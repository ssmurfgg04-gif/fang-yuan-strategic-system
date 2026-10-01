#!/usr/bin/env python3
"""Run ONE shard of the Fang Yuan fidelity benchmark — fully deterministic.

CI-facing sharded runner. No LLM, no network calls: only the deterministic
paths are supported:
  - dynamic        : simulator environments (fixed seeds) + policy compiler
  - canon          : DB FTS retrieval answer + keyword rubric
  - counterfactual : posture-matching rubric + canned/baseline answers

Sharding is stable: item -> int(md5(item_key), 16) % num_shards, so items
land on the same shard regardless of row order or DB re-imports.

Usage:
  python scripts/benchmark/run_shard.py --shard 0 --num-shards 4 \
      --out results/shard_0.json [--layers dynamic,canon,counterfactual] \
      [--db db/fang_yuan_public.db]

Writes the shard JSON, plus a writable COPY of the database
(results/db_shard_<i>.db) into which this shard's benchmark_runs rows are
appended — CI checkouts can be read-only, so run metadata is always
recorded against the copy, which is uploaded as an artifact. If the source
DB itself is writable, rows are also appended there on a best-effort basis
(try/except).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/

from db import db_utils  # noqa: E402

# Reuse the EXACT scoring path of the single-process benchmark runner.
from benchmark.fidelity_benchmark import DIM_TO_F, F_WEIGHTS, score_item  # noqa: E402
from benchmark.adapters import (MockPolicyAgent, RecklessPersonaAgent,  # noqa: E402
                                SafePersonaAgent)

# Layers the deterministic compiler/simulator/adapters can score without any
# LLM. quote_verification / style_concealment are excluded from CI on purpose
# (their meaningful scoring path requires a language model).
DETERMINISTIC_LAYERS = ("dynamic", "canon", "counterfactual")

RUNS_SCHEMA = ("CREATE TABLE IF NOT EXISTS benchmark_runs ("
               "id INTEGER PRIMARY KEY, "
               "run_id TEXT NOT NULL, "
               "item_id INTEGER REFERENCES benchmark_items(id), "
               "model TEXT, adapter TEXT, output_json TEXT, "
               "scores_json TEXT, created_at TEXT)")


def stable_shard(item_key: str, num_shards: int) -> int:
    """md5(item_key) mod N — invariant to row order / DB re-imports."""
    digest = hashlib.md5(item_key.encode("utf-8")).hexdigest()
    return int(digest, 16) % num_shards


def compute_f(dim_scores: dict[str, list[float]]) -> float:
    """Replicate fidelity_benchmark's F aggregation over a subset of items:
    per-dimension mean -> mapped to F letter -> per-letter mean -> weights;
    letters with no items contribute 0.0 (same as the single-process runner).
    """
    f_scores: dict[str, list[float]] = {}
    for dim, scores in dim_scores.items():
        f = DIM_TO_F.get(dim)
        if f:
            f_scores.setdefault(f, []).append(sum(scores) / len(scores))
    f_agg = {k: sum(v) / len(v) for k, v in f_scores.items()}
    return round(sum(F_WEIGHTS[k] * f_agg.get(k, 0.0) for k in F_WEIGHTS), 3)


def load_shard_items(db_path: Path, layers: list[str], shard: int,
                     num_shards: int) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM benchmark_items").fetchall()
    finally:
        conn.close()
    items = []
    for row in rows:
        d = dict(row)
        if d["layer"] not in layers:
            continue
        key = d["item_key"] or f"rowid-{d['id']}"
        if stable_shard(key, num_shards) != shard:
            continue
        d["item_key"] = key
        items.append(d)
    items.sort(key=lambda x: x["item_key"])  # deterministic output order
    return items


def score_items(items: list[dict], agents: list) -> tuple[list[dict], dict, dict]:
    """Score every item with every agent.

    Returns (item_records, summary, latencies). The shard JSON is kept
    byte-deterministic: latency (wall-clock noise) is returned separately so
    it lands only in the DB run rows, never in the JSON.
    """
    records: list[dict] = []
    latencies: dict[str, dict[str, float]] = {}
    per_agent_scores: dict[str, list[float]] = {a.name: [] for a in agents}
    per_agent_dims: dict[str, dict[str, list[float]]] = \
        {a.name: {} for a in agents}
    for item in items:
        rubric = json.loads(item["rubric_json"])
        scores: dict[str, float] = {}
        details: dict[str, object] = {}
        latencies[item["item_key"]] = {}
        for agent in agents:
            d = dict(item)
            if rubric.get("type") == "dynamic":
                d["_adapter_obj"] = agent
            t0 = time.time()
            answer = agent.answer(d)
            score, detail = score_item(d, answer)
            latencies[item["item_key"]][agent.name] = \
                round((time.time() - t0) * 1000, 1)
            scores[agent.name] = score
            details[agent.name] = detail
            per_agent_scores[agent.name].append(score)
            per_agent_dims[agent.name].setdefault(item["dimension"],
                                                  []).append(score)
        records.append({
            "item_key": item["item_key"],
            "id": item["id"],
            "layer": item["layer"],
            "dimension": item["dimension"],
            "holdout": bool(item["holdout"]),
            "scores": scores,
            "details": details,
        })
    summary = {}
    for agent in agents:
        sc = per_agent_scores[agent.name]
        summary[agent.name] = {
            "F": compute_f(per_agent_dims[agent.name]),
            "mean_score": round(sum(sc) / len(sc), 4) if sc else 0.0,
            "n_items": len(sc),
        }
    return records, summary, latencies


def copy_db(src: Path, dst: Path) -> None:
    """Create a writable snapshot of the source DB (sqlite backup API; falls
    back to a raw file copy)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        src_conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        dst_conn = sqlite3.connect(str(dst))
        with dst_conn:
            src_conn.backup(dst_conn)
        src_conn.close()
        dst_conn.close()
    except sqlite3.Error:
        shutil_copy(src, dst)


def shutil_copy(src: Path, dst: Path) -> None:  # pragma: no cover
    import shutil
    shutil.copy2(src, dst)


def append_runs(db_path: Path, run_id: str, records: list[dict],
                agents: dict[str, str],
                latencies: dict[str, dict[str, float]] | None = None) -> bool:
    """Insert benchmark_runs rows. Returns True on success."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(RUNS_SCHEMA)
        key_to_id = {r["item_key"]: r["id"] for r in
                     conn.execute("SELECT id, item_key FROM benchmark_items")}
        for rec in records:
            for agent_name, score in rec["scores"].items():
                detail = dict(rec["details"][agent_name])
                if latencies and agent_name in latencies.get(rec["item_key"], {}):
                    detail["latency_ms"] = latencies[rec["item_key"]][agent_name]
                conn.execute(
                    "INSERT INTO benchmark_runs(run_id, item_id, model, "
                    "adapter, output_json, scores_json, created_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (run_id, key_to_id.get(rec["item_key"]), agent_name,
                     agents[agent_name],
                     json.dumps({"item_key": rec["item_key"],
                                 "layer": rec["layer"],
                                 "dimension": rec["dimension"]},
                                ensure_ascii=False),
                     json.dumps({"score": score, "detail": detail},
                                ensure_ascii=False),
                     db_utils.utcnow()))
        conn.commit()
        return True
    except sqlite3.Error as exc:
        print(f"[run_shard] DB write skipped ({db_path}): {exc}",
              file=sys.stderr)
        return False
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--out", default=None,
                    help="output JSON path (default results/shard_<i>.json)")
    ap.add_argument("--layers",
                    default=",".join(DETERMINISTIC_LAYERS),
                    help="comma-separated subset of: "
                         + ",".join(DETERMINISTIC_LAYERS))
    ap.add_argument("--db", default=None,
                    help="source DB (default db/fang_yuan.db; CI uses "
                         "db/fang_yuan_public.db)")
    args = ap.parse_args()

    if args.num_shards < 1 or args.num_shards > 20:
        ap.error("--num-shards must be in [1, 20]")
    if not 0 <= args.shard < args.num_shards:
        ap.error(f"--shard must be in [0, {args.num_shards - 1}]")
    layers = [x.strip() for x in args.layers.split(",") if x.strip()]
    bad = [x for x in layers if x not in DETERMINISTIC_LAYERS]
    if bad:
        ap.error(
            f"layers {bad} are not deterministically scoreable in CI "
            f"(quote_verification / style_concealment need an LLM adapter); "
            f"allowed: {list(DETERMINISTIC_LAYERS)}")
    if not layers:
        ap.error("no layers requested")

    src_db = Path(args.db).resolve() if args.db else db_utils.DB_PATH
    if not src_db.exists():
        print(f"[run_shard] source DB not found: {src_db}", file=sys.stderr)
        return 2
    # Redirect adapters' DB-backed retrieval (canon layer) to the same DB.
    db_utils.DB_PATH = src_db

    out_path = Path(args.out) if args.out else \
        Path("results") / f"shard_{args.shard}.json"

    agents = [MockPolicyAgent(), SafePersonaAgent(), RecklessPersonaAgent()]
    agent_classes = {a.name: a.__class__.__name__ for a in agents}

    items = load_shard_items(src_db, layers, args.shard, args.num_shards)
    records, summary, latencies = score_items(items, agents)

    result = {
        "schema": "fang_yuan_benchmark_shard_v1",
        "shard": args.shard,
        "num_shards": args.num_shards,
        "layers": layers,
        "agents": [a.name for a in agents],
        "source_db": src_db.name,
        "n_items": len(records),
        "items": records,
        "summary": summary,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    # ---- DB bookkeeping -------------------------------------------------
    run_id = f"shard{args.shard}_of_{args.num_shards}_" \
             f"{'+'.join(layers)}_{int(time.time())}"
    # 1) guaranteed-writable snapshot of the DB + this shard's run rows
    shard_db = out_path.parent / f"db_shard_{args.shard}.db"
    try:
        copy_db(src_db, shard_db)
        ok_copy = append_runs(shard_db, run_id, records, agent_classes,
                              latencies)
    except Exception as exc:  # noqa: BLE001 — never fail scoring on DB copy
        print(f"[run_shard] DB copy/write failed ({shard_db}): {exc}",
              file=sys.stderr)
        ok_copy = False
    # 2) best-effort append to the source DB (skipped when read-only)
    ok_src = append_runs(src_db, run_id, records, agent_classes, latencies)

    print(json.dumps({
        "shard": args.shard,
        "num_shards": args.num_shards,
        "n_items": len(records),
        "out": str(out_path),
        "db_copy": str(shard_db) if ok_copy else None,
        "db_source_appended": ok_src,
        "summary": summary,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
