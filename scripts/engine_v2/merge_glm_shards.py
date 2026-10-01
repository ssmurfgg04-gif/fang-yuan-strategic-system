#!/usr/bin/env python3
"""Merge GLM benchmark shards into a global report + log to DB."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from db import db_utils  # noqa: E402
from benchmark.fidelity_benchmark import DIM_TO_F, F_WEIGHTS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-glob", default="results/glm_shard_*.json")
    ap.add_argument("--out", default="results/glm_merged.json")
    ap.add_argument("--log-db", action="store_true")
    args = ap.parse_args()

    shards = [json.load(open(p)) for p in sorted(Path(".").glob(args.in_glob))]
    if not shards:
        print("no shards found"); sys.exit(1)
    seen, items = set(), []
    for s in shards:
        for it in s["items"]:
            if it["item_key"] in seen:
                continue
            seen.add(it["item_key"])
            items.append(it)
    dim_scores: dict[str, list[float]] = {}
    for it in items:
        dim_scores.setdefault(it["dimension"], []).append(it["score"])
    f_scores: dict[str, list[float]] = {}
    for dim, ss in dim_scores.items():
        f = DIM_TO_F.get(dim)
        if f:
            f_scores.setdefault(f, []).append(sum(ss) / len(ss))
    f_agg = {k: round(sum(v) / len(v), 3) for k, v in f_scores.items()}
    F = round(sum(F_WEIGHTS[k] * f_agg.get(k, 0.0) for k in F_WEIGHTS), 3)
    hold = [it["score"] for it in items if it["holdout"]]
    nonhold = [it["score"] for it in items if not it["holdout"]]

    per_layer: dict[str, dict] = {}
    for it in items:
        per_layer.setdefault(it["layer"], []).append(it["score"])
    layers = {k: round(sum(v) / len(v), 3) for k, v in per_layer.items()}

    report = {
        "run_id": f"glm_selfconsistency_v2_{int(time.time())}",
        "adapter": "glm_selfconsistency_v2",
        "F": F, "dimensions": f_agg, "layers": layers,
        "n_items": len(items), "shards": len(shards),
        "holdout_mean": round(sum(hold) / len(hold), 3) if hold else None,
        "holdout_n": len(hold),
        "nonholdout_mean": round(sum(nonhold) / len(nonhold), 3),
        "timestamp": db_utils.utcnow(),
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.log_db:
        conn = db_utils.connect()
        for it in items:
            row = conn.execute(
                "SELECT id FROM benchmark_items WHERE item_key=?",
                (it["item_key"],)).fetchone()
            if not row:
                continue
            conn.execute(
                "INSERT INTO benchmark_runs(run_id, item_id, model, adapter, "
                "output_json, scores_json, created_at) VALUES (?,?,?,?,?,?,?)",
                (report["run_id"], row["id"], "glm-4.6", "GLMSelfConsistencyAgent",
                 json.dumps({"answer": it["answer_head"]}, ensure_ascii=False),
                 json.dumps({"score": it["score"], "detail": it["detail"]},
                            ensure_ascii=False),
                 db_utils.utcnow()))
        conn.commit()
        print(f"logged {len(items)} runs to DB as {report['run_id']}")


if __name__ == "__main__":
    main()
