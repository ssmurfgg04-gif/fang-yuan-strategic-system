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
                    choices=["single", "full", "deterministic"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--static-mode", default="loop", choices=["loop", "fast"])
    args = ap.parse_args()

    conn = db_utils.connect()
    items = [dict(r) for r in conn.execute(
        "SELECT * FROM benchmark_items ORDER BY id").fetchall()]
    mine = [it for it in items
            if int(hashlib.md5(it["item_key"].encode()).hexdigest(), 16)
            % args.num_shards == args.shard]

    # incremental resume: append-per-item file survives process restarts
    inc_path = Path(args.out.replace(".json", "_items.jsonl"))
    done: dict[str, dict] = {}
    if inc_path.exists():
        for line in inc_path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done[r["item_key"]] = r
            except Exception:
                pass
    todo = [it for it in mine if it["item_key"] not in done]
    print(f"[shard {args.shard}/{args.num_shards}] items={len(mine)} "
          f"done={len(done)} todo={len(todo)}", flush=True)

    from engine_v2.self_consistency import SelfConsistencyLoop
    agent = GLMSelfConsistencyAgent(
        loop=SelfConsistencyLoop(max_workers=args.workers),
        dynamic_mode=args.dynamic_mode, static_mode=args.static_mode)
    inc = inc_path.open("a", encoding="utf-8")
    for n, it in enumerate(todo, 1):
        t0 = time.time()
        try:
            import json as _json
            if _json.loads(it["rubric_json"]).get("type") == "dynamic":
                it["_adapter_obj"] = agent  # score_item dynamic branch needs this
                answer = ""
            else:
                answer = agent.answer(dict(it))
        except Exception as e:  # never let one item kill the shard
            answer = f"__ERROR__ {type(e).__name__}: {e}"
        try:
            score, detail = score_item(dict(it), answer)
        except Exception as e:
            score, detail = 0.0, {"error": str(e)}
        rec = {"item_key": it["item_key"], "layer": it["layer"],
               "dimension": it["dimension"], "holdout": it["holdout"],
               "score": score, "latency_s": round(time.time() - t0, 1),
               "detail": detail, "answer_head": str(answer)[:400]}
        inc.write(json.dumps(rec, ensure_ascii=False) + "\n")
        inc.flush()
        print(f"[shard {args.shard}] {n}/{len(todo)} {it['item_key']} "
              f"score={score} ({rec['latency_s']}s)", flush=True)
    inc.close()

    # aggregate from the incremental file (re-read: includes this run's appends)
    all_done = len(done) + len(todo)
    if all_done < len(mine):
        print(f"[shard {args.shard}] incomplete ({all_done}/{len(mine)}) — resume later")
        return
    done = {}
    for line in inc_path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            done[r["item_key"]] = r
        except Exception:
            pass
    results = [done[it["item_key"]] for it in mine if it["item_key"] in done]
    dim_scores: dict[str, list[float]] = {}
    for r in results:
        dim_scores.setdefault(r["dimension"], []).append(r["score"])

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
