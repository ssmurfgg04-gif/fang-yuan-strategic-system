#!/usr/bin/env python3
"""Sharded LLM-in-the-loop benchmark runner (GLM self-consistency agent).

Usage:
  python scripts/engine_v2/run_glm_shard.py --shard 0 --num-shards 4 --out results/glm_shard_0.json

Sharding: int(md5(item_key),16) % N. Writes per-item scores; does NOT touch the
DB (merge step logs). Deterministic + resumable via the engine_v2 disk caches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from db import db_utils  # noqa: E402
from benchmark.fidelity_benchmark import DIM_TO_F, F_WEIGHTS, score_item  # noqa: E402

from engine_v2.glm_agent import GLMSelfConsistencyAgent  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dynamic-mode", default="single",
                    choices=["single", "full"])
    args = ap.parse_args()

    conn = db_utils.connect()
    items = [dict(r) for r in conn.execute(
        "SELECT * FROM benchmark_items ORDER BY id").fetchall()]
    mine = [it for it in items
            if int(hashlib.md5(it["item_key"].encode()).hexdigest(), 16)
            % args.num_shards == args.shard]
    print(f"[shard {args.shard}/{args.num_shards}] items={len(mine)}", flush=True)

    agent = GLMSelfConsistencyAgent(dynamic_mode=args.dynamic_mode)
    results = []
    dim_scores: dict[str, list[float]] = {}
    for n, it in enumerate(mine, 1):
        t0 = time.time()
        try:
            answer = agent.answer(dict(it))
        except Exception as e:  # never let one item kill the shard
            answer = f"__ERROR__ {type(e).__name__}: {e}"
        try:
            score, detail = score_item(dict(it), answer)
        except Exception as e:
            score, detail = 0.0, {"error": str(e)}
        dim_scores.setdefault(it["dimension"], []).append(score)
        results.append({"item_key": it["item_key"], "layer": it["layer"],
                        "dimension": it["dimension"], "holdout": it["holdout"],
                        "score": score, "latency_s": round(time.time() - t0, 1),
                        "detail": detail,
                        "answer_head": str(answer)[:400]})
        print(f"[shard {args.shard}] {n}/{len(mine)} {it['item_key']} "
              f"score={score} ({results[-1]['latency_s']}s)", flush=True)

    f_scores: dict[str, list[float]] = {}
    for dim, ss in dim_scores.items():
        f = DIM_TO_F.get(dim)
        if f:
            f_scores.setdefault(f, []).append(sum(ss) / len(ss))
    f_agg = {k: round(sum(v) / len(v), 3) for k, v in f_scores.items()}
    F = round(sum(F_WEIGHTS[k] * f_agg.get(k, 0.0) for k in F_WEIGHTS), 3)
    hold = [r["score"] for r in results if r["holdout"]]
    out = {"shard": args.shard, "num_shards": args.num_shards,
           "adapter": agent.name, "F_partial": F, "dimensions": f_agg,
           "n_items": len(results),
           "holdout_mean": round(sum(hold) / len(hold), 3) if hold else None,
           "items": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({"shard": args.shard, "F_partial": F, "n": len(results)}))


if __name__ == "__main__":
    main()
