#!/usr/bin/env python3
"""Fang Yuan DPO training kernel — Qwen2.5-0.5B-Instruct on Kaggle GPU.

Pure torch + transformers + peft (no TRL). Two stages:
  A) SFT with completion-only loss on sft_utility_v1.jsonl (train split)
  B) DPO with precomputed reference logprobs on dpo_pairs_v1.jsonl (train split)

Eval: val-split DPO accuracy, JSON parse rate (greedy), selected-option match.
Outputs to /kaggle/working/out/: adapter_sft/, adapter_dpo/, merged/, metrics.json
"""
from __future__ import annotations

import glob
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

OUT = Path("/kaggle/working/out")
DATA_GLOB = "/kaggle/input/*"
MAXLEN = 1024
BETA = 0.1
LR_SFT = 2e-4
LR_DPO = 5e-5
EPOCHS_SFT = 3
EPOCHS_DPO = 2
SEED = 20260101

random.seed(SEED)
torch.manual_seed(SEED)
SYSTEM_PROMPT = ""


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def find_data_file(name: str) -> Path:
    for p in glob.glob(f"{DATA_GLOB}/**/{name}", recursive=True):
        return Path(p)
    raise FileNotFoundError(name)


def find_base_model() -> str:
    for p in glob.glob(f"{DATA_GLOB}/**/config.json", recursive=True):
        try:
            cfg = json.load(open(p))
            if cfg.get("model_type") == "qwen2" or "qwen" in p.lower():
                return str(Path(p).parent)
        except Exception:
            pass
    return "Qwen/Qwen2.5-0.5B-Instruct"


def load_jsonl(path: Path, split: str | None = None) -> list[dict]:
    recs = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if split is None or r.get("split", "train") == split:
            recs.append(r)
    return recs


def to_ids(tok, x) -> list[int]:
    if isinstance(x, dict):
        x = x["input_ids"]
    return list(x)


# ------------------------------------------------------------------ datasets
class SFTData(Dataset):
    def __init__(self, recs, tok):
        self.ex = []
        for r in recs:
            msgs = r["messages"]
            full = to_ids(tok, tok.apply_chat_template(msgs, tokenize=True))
            prompt = to_ids(tok, tok.apply_chat_template(msgs[:-1], tokenize=True,
                                                         add_generation_prompt=True))
            full = full[:MAXLEN]
            n_prompt = min(len(prompt), len(full))
            labels = [-100] * n_prompt + full[n_prompt:]
            self.ex.append((full, labels))

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


class DPOData(Dataset):
    """Returns (prompt_ids, chosen_ids, rejected_ids)."""

    def __init__(self, recs, tok):
        self.ex = []
        for r in recs:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": r["prompt"]}]
            p = to_ids(tok, tok.apply_chat_template(msgs, tokenize=True,
                                                    add_generation_prompt=True))
            c = tok(r["chosen"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            j = tok(r["rejected"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            self.ex.append((p[: MAXLEN // 2], c[:MAXLEN], j[:MAXLEN]))

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


def collate_pairs(batch, pad_id):
    seqs, ctx = [], []
    for p, c, j in batch:
        seqs.append(c)
        ctx.append((p, "chosen"))
        seqs.append(j)
        ctx.append((p, "rejected"))
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    return ids, mask, ctx


def seq_logprobs(model, ids, mask, prompt_lens):
    """Sum logprob of completion tokens (tokens below prompt_len excluded)."""
    out = model(input_ids=ids, attention_mask=mask)
    logits = out.logits[:, :-1, :]
    targets = ids[:, 1:]
    tmask = mask[:, 1:].clone()
    for i, pl in enumerate(prompt_lens):
        tmask[i, : max(0, pl - 1)] = 0
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_lp = torch.gather(logp, 2, targets.unsqueeze(-1)).squeeze(-1)
    tok_lp = tok_lp * tmask
    return tok_lp.sum(-1)


def collate_sft(batch, pad_id):
    L = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, : len(x)] = torch.tensor(x)
        lab[i, : len(y)] = torch.tensor(y)
        mask[i, : len(x)] = 1
    return ids, lab, mask


def main():
    os.makedirs(OUT, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    model_path = find_base_model()
    log("base model:", model_path)
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = open(find_data_file("system_prompt_v1.txt"), encoding="utf-8").read()

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    sft_train = SFTData(load_jsonl(find_data_file("sft_utility_v1.jsonl"), "train"), tok)
    dpo_train = DPOData(load_jsonl(find_data_file("dpo_pairs_v1.jsonl"), "train"), tok)
    dpo_val = DPOData(load_jsonl(find_data_file("dpo_pairs_v1.jsonl"), "val"), tok)
    log(f"sft_train={len(sft_train)} dpo_train={len(dpo_train)} dpo_val={len(dpo_val)}")

    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(DEV)
    AMP = DEV == "cuda"
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR_SFT, weight_decay=0.0)

    # ------------------------------------------------------------- Stage A SFT
    log("=== Stage A: SFT ===")
    dl = DataLoader(sft_train, batch_size=4, shuffle=True,
                    generator=torch.Generator().manual_seed(SEED),
                    collate_fn=lambda b: collate_sft(b, tok.pad_token_id))
    steps = 0
    t0 = time.time()
    for ep in range(EPOCHS_SFT):
        for ids, lab, mask in dl:
            ids, lab, mask = ids.to(DEV), lab.to(DEV), mask.to(DEV)
            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                out = model(input_ids=ids, attention_mask=mask)
            logits = out.logits[:, :-1, :]
            tgt = lab[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1),
                ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            steps += 1
            if steps % 20 == 0:
                log(f"sft ep{ep} step{steps} loss={loss.item():.4f}")
    for gpr in opt.param_groups:
        gpr["lr"] = LR_DPO
    log(f"SFT done in {time.time() - t0:.0f}s, steps={steps}")
    model.save_pretrained(OUT / "adapter_sft")

    # --------------------------------------------- precompute reference logprobs
    log("=== Reference logprobs (adapter disabled) ===")

    @torch.no_grad()
    def batch_logprobs(ds: DPOData, bs=8, use_ref=True):
        """Returns {(pair_idx, side): sum_logp}."""
        res = {}
        dl = DataLoader(ds, batch_size=bs, shuffle=False,
                        collate_fn=lambda b: collate_pairs(b, tok.pad_token_id))
        idx = 0
        for ids, mask, ctx in dl:
            plens = [len(p) for p, _ in ctx]
            if use_ref:
                with model.disable_adapter():
                    with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                        s = seq_logprobs(model, ids.to(DEV), mask.to(DEV), plens)
            else:
                with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                    s = seq_logprobs(model, ids.to(DEV), mask.to(DEV), plens)
            for k, (p, side) in enumerate(ctx):
                res[(idx + k // 2, side)] = s[k].item()
            idx += len(ctx) // 2
        return res

    t0 = time.time()
    ref_train = batch_logprobs(dpo_train, use_ref=True)
    ref_val = batch_logprobs(dpo_val, use_ref=True)
    log(f"ref logprobs done in {time.time() - t0:.0f}s: "
        f"train={len(ref_train)} val={len(ref_val)}")

    # ------------------------------------------------------------- Stage B DPO
    log("=== Stage B: DPO ===")
    metrics = {"dpo_loss": [], "margins": [], "train_acc": []}
    step = 0
    t0 = time.time()
    for ep in range(EPOCHS_DPO):
        perm = list(torch.randperm(len(dpo_train)).tolist())
        for b0 in range(0, len(perm), 2):
            idxs = perm[b0: b0 + 2]
            batch = [dpo_train[i] for i in idxs]
            ids, mask, ctx = collate_pairs(batch, tok.pad_token_id)
            plens = [len(p) for p, _ in ctx]
            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                s = seq_logprobs(model, ids.to(DEV), mask.to(DEV), plens)
            ref_c = torch.tensor([ref_train[(i, "chosen")] for i in idxs], device=DEV)
            ref_r = torch.tensor([ref_train[(i, "rejected")] for i in idxs], device=DEV)
            pol_c, pol_r = s[0::2], s[1::2]
            margin = (pol_c - ref_c) - (pol_r - ref_r)
            loss = -F.logsigmoid(BETA * margin).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            metrics["dpo_loss"].append(loss.item())
            metrics["margins"].append(margin.mean().item())
            metrics["train_acc"].append((margin > 0).float().mean().item())
            step += 1
            if step % 20 == 0:
                log(f"dpo ep{ep} step{step} loss={loss.item():.4f} "
                    f"margin={margin.mean().item():.3f} "
                    f"acc={sum(metrics['train_acc'][-20:]) / min(20, len(metrics['train_acc'])):.2f}")
    metrics["dpo_steps"] = step
    log(f"DPO done in {time.time() - t0:.0f}s")
    model.save_pretrained(OUT / "adapter_dpo")

    # ----------------------------------------------------------------- Eval
    log("=== Eval on val split ===")
    pol_val = batch_logprobs(dpo_val, use_ref=False)
    wins, n = 0, 0
    for i in range(len(dpo_val)):
        m = (pol_val[(i, "chosen")] - ref_val[(i, "chosen")]) - \
            (pol_val[(i, "rejected")] - ref_val[(i, "rejected")])
        wins += m > 0
        n += 1
    acc = wins / max(1, n)
    log(f"val DPO accuracy: {acc:.3f}")

    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()

    @torch.no_grad()
    def gen_check(n=20):
        ok_json = ok_sel = total = 0
        for r_i, rec in enumerate(load_jsonl(find_data_file("dpo_pairs_v1.jsonl"), "val")[:n]):
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": rec["prompt"]}]
            prompt = to_ids(tok, tok.apply_chat_template(msgs, tokenize=True,
                                                         add_generation_prompt=True))
            ids = torch.tensor([prompt], device=DEV)
            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                out = model.generate(ids, max_new_tokens=480, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            text = tok.decode(out[0][len(prompt):], skip_special_tokens=True)
            total += 1
            try:
                j = json.loads(text.strip())
                ok_json += 1
                gt = json.loads(rec["chosen"])
                if j.get("selected") == gt.get("selected"):
                    ok_sel += 1
            except Exception:
                pass
        return ok_json / max(1, total), ok_sel / max(1, total)

    jr, sr = gen_check()
    log(f"val JSON rate={jr:.2f} selected-match={sr:.2f}")

    metrics["val_dpo_accuracy"] = acc
    metrics["val_json_rate"] = jr
    metrics["val_selected_match"] = sr
    metrics["final_dpo_loss_mean"] = sum(metrics["dpo_loss"][-10:]) / max(1, len(metrics["dpo_loss"][-10:]))
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    log("=== Merging adapter for export ===")
    merged = model.merge_and_unload()
    merged = merged.to(torch.float16)
    merged.save_pretrained(OUT / "merged", safe_serialization=True)
    tok.save_pretrained(OUT / "merged")
    log("ALL DONE")
    print("RESULT_JSON", json.dumps({"val_dpo_accuracy": acc, "val_json_rate": jr,
                                     "val_selected_match": sr, "dpo_steps": step}))


if __name__ == "__main__":
    main()
