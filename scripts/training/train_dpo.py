#!/usr/bin/env python3
"""Universal QLoRA/DPO trainer for the Fang Yuan policy surgery.

Works on any instruction model from 0.5B to 12B+ (tested path: Qwen2.5 family,
also fits Gemma-3 / Llama-3.2). One script, two stages:

  Stage A (optional, --stage sft|both): SFT on sft_decisions.jsonl
          -> teaches the decision-JSON schema.
  Stage B (--stage dpo|both): DPO on dpo_pairs.jsonl
          -> teaches the trade-off weights (cold-optimal vs safe/moral vs
             reckless vs sunk-cost). This is the stage that moves benchmark F.

Size presets are applied automatically from the model name and can be
overridden by flags. fp16/bf16 is auto-detected (Kaggle T4 = fp16, L4/A100 =
bf16). Models >= 3B default to 4-bit NF4 QLoRA.

Examples
--------
# 0.5B on a free Kaggle T4 (minutes)
python scripts/training/train_dpo.py --base Qwen/Qwen2.5-0.5B-Instruct --out runs/fy-0.5b

# 7B, 4-bit, T4 x2
python scripts/training/train_dpo.py --base Qwen/Qwen2.5-7B-Instruct --out runs/fy-7b --merge

# Multi-LoRA: specialized concealment adapter
python scripts/training/train_dpo.py --base Qwen/Qwen2.5-3B-Instruct \
    --task-filter concealment --stage dpo --out runs/fy-3b-concealment

# push finished adapter to the Hugging Face Hub (needs HF_TOKEN env var)
python scripts/training/train_dpo.py --base Qwen/Qwen2.5-0.5B-Instruct \
    --out runs/fy-0.5b --push-hf <your-name>/fangyuan-0.5b-lora
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

PRESETS = {  # key: size tag parsed from model name
    "0.5B": dict(r=16, alpha=32, lr=2e-4, bsz=8, ga=2, seq=1024,
                 quant=False, ep_sft=3, ep_dpo=2),
    "1.5B": dict(r=16, alpha=32, lr=1e-4, bsz=4, ga=4, seq=1024,
                 quant=False, ep_sft=2, ep_dpo=2),
    "3B": dict(r=16, alpha=32, lr=1e-4, bsz=4, ga=4, seq=1536,
               quant=True, ep_sft=2, ep_dpo=2),
    "7B": dict(r=16, alpha=32, lr=8e-5, bsz=2, ga=8, seq=1536,
               quant=True, ep_sft=2, ep_dpo=2),
    "12B": dict(r=8, alpha=16, lr=6e-5, bsz=1, ga=16, seq=2048,
                quant=True, ep_sft=1, ep_dpo=2),
}


def detect_size(model_name: str) -> str:
    m = re.search(r"(\d+(?:\.\d+)?)b", model_name.lower())
    if not m:
        return "7B"
    p = float(m.group(1))
    if p <= 0.8:
        return "0.5B"
    if p <= 1.6:
        return "1.5B"
    if p <= 3.5:
        return "3B"
    if p <= 8:
        return "7B"
    return "12B"


def load_jsonl(path: Path, keys: list[str]) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({k: r[k] for k in keys if k in r})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default=str(REPO / "data/training/dpo_pairs.jsonl"))
    ap.add_argument("--sft-data", default=str(REPO / "data/training/sft_decisions.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=["both", "sft", "dpo"], default="both")
    ap.add_argument("--task-filter", default="",
                    help="train only on one task: policy|concealment|quotes|canon (Multi-LoRA)")
    ap.add_argument("--lora-r", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--bsz", type=int, default=0)
    ap.add_argument("--ga", type=int, default=0)
    ap.add_argument("--seq", type=int, default=0)
    ap.add_argument("--epochs-sft", type=int, default=0)
    ap.add_argument("--epochs-dpo", type=int, default=0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--4bit", dest="quant", action="store_true", default=None)
    ap.add_argument("--no-4bit", dest="quant", action="store_false")
    ap.add_argument("--merge", action="store_true", help="also save merged full-precision model")
    ap.add_argument("--push-hf", default="", help="HF repo id to upload the adapter (uses HF_TOKEN)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        import inspect

        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import trl
        from trl import DPOConfig, DPOTrainer
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        print(f"[deps] missing: {e}\n"
              "pip install -U 'transformers>=4.44,<5' 'trl>=0.12,<0.20' "
              "'peft>=0.12' 'accelerate>=0.33' 'datasets>=2.20' bitsandbytes")
        return 2

    size = detect_size(args.base)
    p = dict(PRESETS[size])
    if args.quant is not None:
        p["quant"] = args.quant
    lora_r = args.lora_r or p["r"]
    lr = args.lr or p["lr"]
    bsz = args.bsz or p["bsz"]
    ga = args.ga or p["ga"]
    seq = args.seq or p["seq"]
    ep_sft = args.epochs_sft or p["ep_sft"]
    ep_dpo = args.epochs_dpo or p["ep_dpo"]

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
    print(f"[cfg] base={args.base} size={size} quant4bit={p['quant']} "
          f"dtype={'bf16' if bf16_ok else 'fp16'} r={lora_r} lr={lr} "
          f"bsz={bsz} ga={ga} seq={seq} ep_sft={ep_sft} ep_dpo={ep_dpo}")

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        print("[warn] tokenizer has no chat template; DPO pairs assume one.")

    quant_cfg = None
    if p["quant"]:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype)

    model_kwargs = dict(quantization_config=quant_cfg) if quant_cfg else {}
    model_kwargs.update(torch_dtype=compute_dtype, device_map="auto",
                        trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, **model_kwargs)
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_r * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])

    common = dict(output_dir=args.out, per_device_train_batch_size=bsz,
                  gradient_accumulation_steps=ga, gradient_checkpointing=True,
                  gradient_checkpointing_kwargs={"use_reentrant": False},
                  logging_steps=10, save_strategy="no", seed=args.seed,
                  bf16=bf16_ok, fp16=not bf16_ok, report_to=[])

    cur_out = Path(args.out)

    # ------------------------------------------------ Stage A: SFT (schema)
    # TRL renamed max_seq_length -> max_length across versions; detect once.
    sft_len_key = ("max_length" if "max_length" in
                   inspect.signature(SFTConfig.__init__).parameters
                   else "max_seq_length")
    if args.stage in ("both", "sft"):
        sft_path = Path(args.sft_data)
        if sft_path.exists():
            rows = load_jsonl(sft_path, ["messages"])
            ds = Dataset.from_list(rows)
            cfg = SFTConfig(**{sft_len_key: seq}, num_train_epochs=ep_sft,
                            learning_rate=lr, **common)
            trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                                 processing_class=tok, peft_config=peft_cfg)
            trainer.train()
            sft_dir = cur_out / "stage_a_sft"
            trainer.save_model(str(sft_dir))
            tok.save_pretrained(str(sft_dir))
            print(f"[ok] Stage A saved -> {sft_dir}")
            # continue Stage B from the SFT adapter on a fresh base load
            del model, trainer
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = PeftModel.from_pretrained(
                AutoModelForCausalLM.from_pretrained(args.base, **model_kwargs),
                str(sft_dir), is_trainable=True)
            model.config.use_cache = False
        else:
            print(f"[skip] no SFT data at {sft_path}; DPO from base")

    # ------------------------------------------------ Stage B: DPO (weights)
    if args.stage in ("both", "dpo"):
        rows = load_jsonl(Path(args.data),
                          ["task", "system", "prompt", "chosen", "rejected"])
        if args.task_filter:
            rows = [r for r in rows if r.get("task") == args.task_filter]
            print(f"[cfg] task filter '{args.task_filter}': {len(rows)} pairs")
        conv = []
        for r in rows:
            conv.append({
                "prompt": [{"role": "system", "content": r["system"]},
                           {"role": "user", "content": r["prompt"]}],
                "chosen": [{"role": "assistant", "content": r["chosen"]}],
                "rejected": [{"role": "assistant", "content": r["rejected"]}],
            })
        ds = Dataset.from_list(conv)
        dpo_lr = lr * 0.5  # DPO wants a gentler step than SFT
        cfg = DPOConfig(beta=args.beta, max_length=seq, max_prompt_length=seq // 2,
                        num_train_epochs=ep_dpo, learning_rate=dpo_lr, **common)
        trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds,
                             processing_class=tok, peft_config=peft_cfg)
        trainer.train()
        dpo_dir = cur_out / "stage_b_dpo"
        trainer.save_model(str(dpo_dir))
        tok.save_pretrained(str(dpo_dir))
        print(f"[ok] Stage B adapter saved -> {dpo_dir}")
        cur_out = dpo_dir

    # ------------------------------------------------------- optional merge
    if args.merge:
        try:
            from peft import PeftModel
            merged_dir = cur_out.parent / "merged"
            base = AutoModelForCausalLM.from_pretrained(
                args.base, torch_dtype=torch.float16,
                low_cpu_mem_usage=True, trust_remote_code=True)
            peft_model = PeftModel.from_pretrained(base, str(cur_out))
            merged = peft_model.merge_and_unload()
            merged.save_pretrained(str(merged_dir))
            tok.save_pretrained(str(merged_dir))
            print(f"[ok] merged fp16 model -> {merged_dir}")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] merge failed ({e}); adapter-only output stands.")

    # -------------------------------------------------------- optional push
    if args.push_hf:
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("[warn] HF_TOKEN not set; skipping Hub upload.")
        else:
            from huggingface_hub import HfApi
            HfApi(token=token).upload_folder(
                folder_path=str(cur_out), repo_id=args.push_hf,
                repo_type="model", exist_ok=True)
            print(f"[ok] pushed adapter -> https://huggingface.co/{args.push_hf}")

    print("\n[next] serve + score against the benchmark:")
    print(f"  1) merge: python {__file__} --help  (or use --merge)")
    print(f"  2) llama-server -m <merged-or-gguf> --port 8080 -c 4096")
    print(f"  3) python -m scripts.benchmark.fidelity_benchmark --base-url http://127.0.0.1:8080/v1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
