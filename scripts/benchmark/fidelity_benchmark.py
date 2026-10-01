#!/usr/bin/env python3
"""Run the Fang Yuan Fidelity Benchmark against adapters and store scores.

F = 0.20*O + 0.15*A + 0.15*R + 0.15*I + 0.15*P + 0.10*M + 0.10*C
  O objective_fixation        A attachmentlessness      R risk_calibration
  I information/reality       P path_optimization       M method_flexibility
  C concealment_discipline
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import db_utils  # noqa: E402
from simulator.environments import ENVS  # noqa: E402

from .adapters import (LlamaCppAdapter, MockPolicyAgent, RecklessPersonaAgent,
                       SafePersonaAgent)  # noqa: E402

DIM_TO_F = {
    "objective_fixation": "O",
    "attachmentlessness": "A",
    "risk_calibration": "R",
    "reality_recognition": "I",
    "information_discipline": "I",
    "canon": "I",
    "path_optimization": "P",
    "method_flexibility": "M",
    "concealment_discipline": "C",
}
F_WEIGHTS = {"O": 0.20, "A": 0.15, "R": 0.15, "I": 0.15, "P": 0.15,
             "M": 0.10, "C": 0.10}

GENERIC_ASSISTANT_MARKERS = ("it depends", "pros and cons", "weigh the options",
                             "proceed cautiously", "careful analysis")


def extract_posture(text: str) -> str | None:
    t = text.lower()
    for marker, posture in [
            ("posture: direct", "direct"), ("posture: indirect", "indirect"),
            ("posture: trade", "trade_or_alliance"),
            ("posture: investigate", "investigate"),
            ("posture: retreat", "retreat"),
            ("strike now", "direct"), ("attack directly", "direct"),
            ("accept the gain", "direct"),
            ("abandon", "retreat"), ("decline", "retreat"),
            ("dual-source", "investigate"), ("verify", "investigate"),
            ("investigate", "investigate"),
            ("swap", "trade_or_alliance"), ("secure recurring", "trade_or_alliance"),
            ("restructure", "trade_or_alliance"),
            ("guide", "indirect"), ("sabotage", "indirect"),
            ("sincere incompetence", "indirect"),
            ("small contract", "investigate")]:
        if marker in t:
            return posture
    return None


def score_item(item: dict, answer: str) -> tuple[float, dict]:
    """Return (score 0..1, detail)."""
    rubric = json.loads(item["rubric_json"])
    rtype = rubric.get("type")
    detail: dict = {}

    if rtype == "keyword_expect":
        text = answer.lower()
        hits = sum(1 for k in rubric["expect"] if k.lower() in text)
        need = rubric.get("min_hits", 1)
        detail = {"hits": hits, "need": need}
        return (min(1.0, hits / max(need, 1)), detail)

    if rtype == "posture_match":
        posture = extract_posture(answer)
        best = rubric.get("best_posture")
        # generic-assistant non-decisions score zero on decision consistency
        if posture is None and any(m in answer.lower()
                                   for m in GENERIC_ASSISTANT_MARKERS):
            return 0.0, {"posture": None, "reason": "no_decision"}
        score = 1.0 if posture == best else (0.4 if posture else 0.0)
        return score, {"posture": posture, "expected": best}

    if rtype == "quote_verdict":
        text = answer.lower()
        trap = rubric.get("trap")
        if rubric.get("verdict") == "unverifiable":
            good = any(m in text for m in
                       ["not verified", "cannot verify", "unverified",
                        "not confirmed in the current corpus"])
            fabrication = any(m in text for m in
                              ["verbatim", "trust me", "exact chinese is",
                               "「"]) and not good
            return (1.0 if good else 0.0), {"good": good, "fabrication": fabrication}
        # classic attribution
        work = rubric.get("verdict", "").replace("classic_", "")
        names = {"sunzi": ["孫子", "sunzi", "art of war"],
                 "daodejing": ["道德經", "daodejing", "tao te"],
                 "shiji": ["史記", "shiji", "貨殖"],
                 "hanfeizi": ["韓非", "hanfei"]}
        named = any(m in text for m in names.get(work, []))
        not_fy = ("not fang yuan" in text or "chinese strategic tradition" in text
                  or "attribution" in text)
        return (1.0 if (named and not_fy) else 0.0,
                {"named_work": named, "denied_canon": not_fy})

    if rtype == "mask_test":
        text = answer.lower()
        conceal = rubric.get("expect_conceal")
        leaks = [m for m in ["eternal life", "demonic path", "true objective",
                             "ultimate goal is"] if m in text]
        length = len(answer.split())
        if conceal:
            ok = not leaks and length <= 60
            return (1.0 if ok else (0.5 if not leaks else 0.0)), \
                {"leaks": leaks, "words": length}
        ok = "contingency" in text or "withdraw" in text or "operationally" in text
        return (1.0 if ok else 0.6), {"shared_contingency": ok}

    if rtype == "dynamic":
        env_cls = ENVS[rubric["env"]]
        adapter_obj = item["_adapter_obj"]
        env = env_cls(seed=rubric["seed"])
        traj = env.run(_DynamicAgentWrapper(adapter_obj))
        m = env.metrics()
        survival_ok = (not rubric.get("min_survival")) or m.survival
        options_ok = m.future_options_remaining >= rubric.get("min_options", 0)
        score = (1.0 if survival_ok else 0.0) * 0.6 + \
                (m.future_options_remaining / 3.0) * 0.25 + \
                min(1.0, max(0.0, m.final_resources) / 80.0) * 0.15
        return round(score, 3), {"metrics": m.as_dict(),
                                 "survival_ok": survival_ok,
                                 "options_ok": options_ok}

    return 0.0, {"reason": "unknown rubric type"}


class _DynamicAgentWrapper:
    def __init__(self, adapter):
        self.adapter = adapter

    def choose(self, obs, legal_actions):
        return self.adapter.choose_dynamic(obs, legal_actions)


def run(adapter, run_id: str | None = None):
    conn = db_utils.connect()
    db_utils.init_schema(conn)
    items = conn.execute(
        "SELECT * FROM benchmark_items ORDER BY id").fetchall()
    run_id = run_id or f"{adapter.name}_{int(time.time())}"
    dim_scores: dict[str, list[float]] = {}
    for item in items:
        d = dict(item)
        if json.loads(d["rubric_json"])["type"] == "dynamic":
            d["_adapter_obj"] = adapter
        t0 = time.time()
        answer = adapter.answer(d)
        score, detail = score_item(d, answer)
        dim = d["dimension"]
        dim_scores.setdefault(dim, []).append(score)
        conn.execute(
            "INSERT INTO benchmark_runs(run_id, item_id, model, adapter, "
            "output_json, scores_json, created_at) VALUES (?,?,?,?,?,?,?)",
            (run_id, d["id"], adapter.name, adapter.__class__.__name__,
             json.dumps({"answer": answer[:2000]}, ensure_ascii=False),
             json.dumps({"score": score, "detail": detail,
                         "latency_ms": round((time.time() - t0) * 1000, 1)},
                        ensure_ascii=False),
             db_utils.utcnow()))
    conn.commit()

    # aggregate per F-dimension
    f_scores = {}
    for dim, scores in dim_scores.items():
        f = DIM_TO_F.get(dim)
        if f:
            f_scores.setdefault(f, []).append(sum(scores) / len(scores))
    f_agg = {k: round(sum(v) / len(v), 3) for k, v in f_scores.items()}
    F = round(sum(F_WEIGHTS[k] * f_agg.get(k, 0.0) for k in F_WEIGHTS), 3)
    summary = {"run_id": run_id, "adapter": adapter.name, "F": F,
               "dimensions": f_agg, "n_items": len(items)}

    # holdout-only score
    hold = conn.execute(
        """SELECT br.scores_json FROM benchmark_runs br
           JOIN benchmark_items bi ON bi.id = br.item_id
           WHERE br.run_id=? AND bi.holdout=1""", (run_id,)).fetchall()
    if hold:
        hs = [json.loads(h["scores_json"])["score"] for h in hold]
        summary["holdout_mean"] = round(sum(hs) / len(hs), 3)
        summary["holdout_n"] = len(hs)
    return summary, conn


def main():
    adapters = [MockPolicyAgent(), SafePersonaAgent(), RecklessPersonaAgent()]
    if len(sys.argv) > 2 and sys.argv[1] == "--base-url":
        adapters.append(LlamaCppAdapter(sys.argv[2]))
    results = []
    for ad in adapters:
        summary, conn = run(ad)
        results.append(summary)
        print(json.dumps(summary, ensure_ascii=False))
    out = Path(__file__).resolve().parents[2] / "docs" / "benchmark_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
