#!/usr/bin/env python3
"""Fang Yuan DPO eval kernel — runs on Kaggle GPU against the merged model.

Mounts:
  /kaggle/input/fy-dpo-train-v1/merged   (training kernel output)
  /kaggle/input/fy-dpo-v1/...            (dataset: dpo_pairs_v1.jsonl etc.)

Evaluates ALL val items: JSON parse rate (fence-stripped), selected-option
match, posture accuracy on counterfactual items, and logs sample generations.
"""
from __future__ import annotations

import glob
import json
import os
import re
import time
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = Path("/kaggle/working/out")
POSTURES = ["direct_action", "indirect_action", "trade_or_alliance",
            "investigation_or_delay", "retreat_or_abandonment"]
POSTURE_ALIASES = {"retreat": "retreat_or_abandonment",
                   "investigate": "investigation_or_delay",
                   "direct": "direct_action", "indirect": "indirect_action"}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def find_file(name: str) -> Path:
    for p in glob.glob(f"/kaggle/input/**/{name}", recursive=True):
        return Path(p)
    raise FileNotFoundError(name)


def find_merged() -> str:
    for pat in ["/kaggle/input/fy-dpo-train-v1/merged",
                "/kaggle/input/*/merged",
                "/kaggle/input/**/merged"]:
        hits = glob.glob(pat)
        if hits and (Path(hits[0]) / "config.json").exists():
            return hits[0]
    raise FileNotFoundError("merged model")


def strip_fences(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # tolerate leading prose before the first brace
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j > i:
        return text[i:j + 1]
    return text


def extract_posture(text: str) -> str | None:
    t = text.lower()
    for marker, posture in [
            ("posture: direct", "direct"), ("posture: indirect", "indirect"),
            ("posture: trade", "trade_or_alliance"),
            ("posture: investigate", "investigate"),
            ("posture: retreat", "retreat")]:
        if marker in t:
            return posture
    return None


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

    system_prompt = open(find_file("system_prompt_v1.txt"),
                         encoding="utf-8").read()
    val = []
    for line in open(find_file("dpo_pairs_v1.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r.get("split") == "val":
            val.append(r)
    log(f"val items: {len(val)}")

    ok_json = ok_sel = 0
    cf_total = cf_ok = 0
    samples = []
    for i, r in enumerate(val):
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": r["prompt"]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt",
                  add_special_tokens=False)["input_ids"].to("cuda")
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=480, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        raw = tok.decode(out[0][len(ids[0]):], skip_special_tokens=True)
        clean = strip_fences(raw)
        parsed = None
        try:
            parsed = json.loads(clean)
            ok_json += 1
            gt = None
            try:
                gt = json.loads(r["chosen"])
            except Exception:
                pass
            if gt and parsed.get("selected") == gt.get("selected"):
                ok_sel += 1
        except Exception:
            pass
        # posture accuracy on counterfactual pairs
        if r.get("pair_type") == "cf_adaptation_vs_recitation":
            cf_total += 1
            best = (r.get("meta") or {}).get("best_posture")
            best = POSTURE_ALIASES.get(best, best)
            picked = extract_posture(raw)
            if parsed:
                sel = parsed.get("selected", "")
                opt = next((o for o in parsed.get("options", [])
                            if o.get("name") == sel), None)
                picked = (opt or {}).get("posture") or picked
            if picked and best and picked.startswith(best.split("_")[0]) or picked == best:
                cf_ok += 1
        if i < 3:
            samples.append({"item": i, "pair_type": r.get("pair_type"),
                            "gen_head": raw[:300]})
        if i % 10 == 0:
            log(f"eval {i}/{len(val)}")

    report = {
        "val_items": len(val),
        "json_rate": round(ok_json / max(1, len(val)), 3),
        "selected_match": round(ok_sel / max(1, len(val)), 3),
        "cf_posture_accuracy": round(cf_ok / max(1, cf_total), 3)
        if cf_total else None,
        "cf_total": cf_total,
        "samples": samples,
    }
    (OUT / "eval_metrics.json").write_text(json.dumps(report, indent=2))
    log("RESULT_JSON", json.dumps(report))


if __name__ == "__main__":
    main()
