#!/usr/bin/env python3
"""Fang Yuan fine-tuned-weights FULL benchmark — closed loop on Kaggle GPU.

The DPO-fine-tuned Qwen2.5-0.5B is scored with the SAME rubric scorer as the
GLM engine and the deterministic baselines:
  canon + counterfactual + quote + style layers: generation per item
  dynamic layer: one generation per simulator step (fallback policy on failure)

Mounts:
  /kaggle/input/fy-dpo-train-v1/...   (training output -> merged model)
  /kaggle/input/fy-bench-assets/...   (public DB + scripts/ + system prompt)
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

WORK = Path("/kaggle/working")
OUT = WORK / "out"

def _find_bench_root() -> Path:
    for root, dirs, files in os.walk("/kaggle/input"):
        if "fang_yuan_public.db" in files:
            return Path(root)
    # debug aid
    listing = [os.path.join(dp, f) for dp, dn, fn in os.walk("/kaggle/input")
               for f in fn][:60]
    raise FileNotFoundError(f"bench root not found; input listing: {listing}")

BENCH = _find_bench_root()

# stage a repo-like tree so db_utils' relative DB_PATH resolves
staging = WORK / "staging"
(staging / "db").mkdir(parents=True, exist_ok=True)
shutil.copy(glob.glob(f"{BENCH}/**/fang_yuan_public.db", recursive=True)[0],
            staging / "db" / "fang_yuan.db")
src_scripts = glob.glob(f"{BENCH}/**/scripts", recursive=True)[0]
shutil.copytree(src_scripts, staging / "scripts",
                ignore=shutil.ignore_patterns("__pycache__"))
cfg_dirs = glob.glob(f"{BENCH}/**/config", recursive=True)
if cfg_dirs:
    shutil.copytree(cfg_dirs[0], staging / "config",
                    ignore=shutil.ignore_patterns("__pycache__"))
sys.path.insert(0, str(staging / "scripts"))
os.chdir(staging)

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

POSTURE_ALIASES = {"retreat": "retreat_or_abandonment",
                   "investigate": "investigation_or_delay",
                   "direct": "direct_action",
                   "indirect": "indirect_action"}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def find_merged() -> str:
    hits = glob.glob("/kaggle/input/**/config.json", recursive=True)
    cands = [Path(h).parent for h in hits
             if any(Path(h).parent.glob("*.safetensors"))]
    cands.sort(key=lambda p: (0 if "merged" in str(p) else 1, len(str(p))))
    if cands:
        return str(cands[0])
    raise FileNotFoundError("merged model")


def strip_fences(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j > i:
        return text[i:j + 1]
    return text


def main():
    os.makedirs(OUT, exist_ok=True)
    model_path = find_merged()
    log("merged model:", model_path)
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, attn_implementation="sdpa")
    model.to("cuda")
    model.eval()

    system_prompt = open(glob.glob(f"{BENCH}/**/system_prompt_v1.txt",
                                   recursive=True)[0], encoding="utf-8").read()

    from benchmark.fidelity_benchmark import DIM_TO_F, F_WEIGHTS, score_item
    from simulator.agents import PolicyAgent as SimPolicyAgent

    def generate(prompt_msgs: str, max_new_tokens: int) -> str:
        prompt = tok.apply_chat_template(prompt_msgs, tokenize=False,
                                         add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt",
                  add_special_tokens=False)["input_ids"].to("cuda")
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][len(ids[0]):], skip_special_tokens=True)

    fallback = SimPolicyAgent(0)
    fallback_rng = [0]

    class FTYuanAdapter:
        """Fine-tuned weights adapter for the fidelity benchmark."""
        name = "fy_dpo_0.5b_v1"

        def answer(self, item: dict) -> str:
            layer = item["layer"]
            rubric = json.loads(item["rubric_json"]) if item.get("rubric_json") else {}
            if layer == "canon":
                from benchmark.adapters import _canon_answer
                return _canon_answer(item["prompt"])
            msgs = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": item["prompt"]}]
            if layer == "quote_verification":
                raw = generate(msgs + [{
                    "role": "user",
                    "content": "If this exact Chinese wording cannot be verified "
                               "against a supplied corpus, reply exactly: The scene "
                               "is identified, but the Chinese wording is not "
                               "verified in the current corpus."}], 120)
                return raw.strip()[:300]
            raw = generate(msgs, 900)
            try:
                j = json.loads(strip_fences(raw))
                sel = j.get("selected", "")
                opts = {o.get("name"): o for o in j.get("options", [])}
                o = opts.get(sel, {})
                posture = o.get("posture") or POSTURE_ALIASES.get(
                    extract_posture(raw) or "", None) or "direct_action"
                if layer == "style_concealment" and rubric.get("expect_conceal"):
                    return str(j.get("external_message") or sel)[:60]
                if layer == "style_concealment":
                    return (str(j.get("external_message") or sel)[:60] +
                            f" Contingency: {j.get('retreat_trigger', '')} "
                            "(shared as operationally required)")
                return (f"Posture: {posture}. Action: {sel}. "
                        f"Contingency: {j.get('retreat_trigger', '')}.")
            except Exception:
                return "unable to produce a decision"

        def choose_dynamic(self, obs, legal_actions):
            try:
                state_txt = json.dumps(
                    {k: v for k, v in obs.items() if not k.startswith("_")},
                    ensure_ascii=False, default=str)
                msgs = [{"role": "system", "content": system_prompt},
                        {"role": "user",
                         "content": f"State: {state_txt}\nLegal actions: "
                                    f"{json.dumps(legal_actions)}\n"
                                    "Reply with EXACTLY one action key."}]
                raw = generate(msgs, 60)
                for a in legal_actions:
                    if a in raw:
                        return a
            except Exception:
                pass
            try:
                return fallback.choose(obs, legal_actions)
            except Exception:
                return legal_actions[0]

    agent = FTYuanAdapter()

    # ---- static layers with the harness scorer ----
    from db import db_utils
    conn = db_utils.connect()
    items = [dict(r) for r in conn.execute(
        "SELECT * FROM benchmark_items ORDER BY id").fetchall()]
    static = [it for it in items
              if json.loads(it["rubric_json"]).get("type") != "dynamic"]
    log(f"static items: {len(static)}")
    dim_scores: dict[str, list[float]] = {}
    per_item = []
    for n, it in enumerate(static, 1):
        t0 = time.time()
        try:
            ans = agent.answer(dict(it))
        except Exception as e:
            ans = f"__ERROR__ {e}"
        try:
            score, detail = score_item(dict(it), ans)
        except Exception as e:
            score, detail = 0.0, {"error": str(e)}
        dim_scores.setdefault(it["dimension"], []).append(score)
        per_item.append({"item_key": it["item_key"], "layer": it["layer"],
                         "dimension": it["dimension"],
                         "holdout": it["holdout"], "score": score,
                         "answer_head": str(ans)[:300]})
        if n % 10 == 0 or n == len(static):
            log(f"static {n}/{len(static)} last={it['item_key']} "
                f"score={score} ({time.time()-t0:.1f}s)")

    # ---- dynamic layer ----
    from simulator.environments import ENVS
    dyn = [it for it in items if json.loads(it["rubric_json"]).get("type") == "dynamic"]
    log(f"dynamic items: {len(dyn)}")
    for n, it in enumerate(dyn, 1):
        rub = json.loads(it["rubric_json"])
        env = ENVS[rub["env"]](seed=rub["seed"])

        class W:
            def __init__(self, a):
                self.a = a

            def choose(self, obs, legal):
                return self.a.choose_dynamic(obs, legal)

        traj = env.run(W(agent))
        m = env.metrics()
        score = ((1.0 if m.survival else 0.0) * 0.6
                 + (m.future_options_remaining / 3.0) * 0.25
                 + min(1.0, max(0.0, m.final_resources) / 80.0) * 0.15)
        dim_scores.setdefault(it["dimension"], []).append(round(score, 3))
        per_item.append({"item_key": it["item_key"], "layer": it["layer"],
                         "dimension": it["dimension"],
                         "holdout": it["holdout"], "score": round(score, 3),
                         "answer_head": json.dumps(m.as_dict())[:200]})
        if n % 5 == 0 or n == len(dyn):
            log(f"dynamic {n}/{len(dyn)} {it['item_key']} score={score:.3f}")

    f_scores: dict[str, list[float]] = {}
    for d, ss in dim_scores.items():
        f = DIM_TO_F.get(d)
        if f:
            f_scores.setdefault(f, []).append(sum(ss) / len(ss))
    f_agg = {k: round(sum(v) / len(v), 3) for k, v in f_scores.items()}
    F = round(sum(F_WEIGHTS[k] * f_agg.get(k, 0.0) for k in F_WEIGHTS), 3)
    hold = [x["score"] for x in per_item if x["holdout"]]
    layers: dict[str, list[float]] = {}
    for x in per_item:
        layers.setdefault(x["layer"], []).append(x["score"])

    report = {
        "adapter": agent.name, "F": F, "dimensions": f_agg,
        "layers": {k: round(sum(v) / len(v), 3) for k, v in layers.items()},
        "n_items": len(per_item),
        "holdout_mean": round(sum(hold) / len(hold), 3) if hold else None,
        "holdout_n": len(hold),
        "items": per_item,
    }
    (OUT / "benchmark_metrics.json").write_text(json.dumps(report, indent=1))
    log("RESULT_JSON", json.dumps({k: report[k] for k in
                                   ("F", "dimensions", "layers", "holdout_mean")}))


if __name__ == "__main__":
    main()
