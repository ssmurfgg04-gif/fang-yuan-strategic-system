#!/usr/bin/env python3
"""Merge sharded benchmark JSONs into a global report.

Reads every shard_*.json produced by run_shard.py (recursively under --in),
validates shard completeness + item disjointness, recomputes global
per-agent F (same aggregation as fidelity_benchmark), per-layer breakdowns,
and prints a markdown summary table to stdout.

Usage:
  python scripts/benchmark/merge_shards.py --in results \
      --out results/merged_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/

from benchmark.fidelity_benchmark import DIM_TO_F, F_WEIGHTS  # noqa: E402


def compute_f(dim_scores: dict[str, list[float]]) -> float:
    """Same aggregation as fidelity_benchmark.run() over a set of items."""
    f_scores: dict[str, list[float]] = {}
    for dim, scores in dim_scores.items():
        f = DIM_TO_F.get(dim)
        if f:
            f_scores.setdefault(f, []).append(sum(scores) / len(scores))
    f_agg = {k: sum(v) / len(v) for k, v in f_scores.items()}
    return round(sum(F_WEIGHTS[k] * f_agg.get(k, 0.0) for k in F_WEIGHTS), 3)


def load_shards(indir: Path) -> list[dict]:
    # accept both shard_3.json and shard3.json naming
    files: set[Path] = set()
    for pattern in ("shard_*.json", "shard[0-9]*.json"):
        files.update(indir.rglob(pattern))
    files = {f for f in files
             if f.name not in ("merged_benchmark.json",)}  # never re-ingest
    if not files:
        raise SystemExit(f"[merge_shards] no shard_*.json files under {indir}")
    shards: list[dict] = []
    for f in sorted(files):
        data = json.loads(f.read_text(encoding="utf-8"))
        if "shard" not in data or "num_shards" not in data:
            raise SystemExit(f"[merge_shards] {f} is not a shard JSON")
        shards.append(data)
    num_shards = {s["num_shards"] for s in shards}
    if len(num_shards) != 1:
        raise SystemExit(f"[merge_shards] mixed num_shards: {sorted(num_shards)}")
    n = num_shards.pop()
    have = sorted(s["shard"] for s in shards)
    if len(set(have)) != len(have):
        raise SystemExit(f"[merge_shards] duplicate shard indices: {have}")
    missing = sorted(set(range(n)) - set(have))
    if missing:
        raise SystemExit(f"[merge_shards] missing shards {missing} "
                         f"(have {have} of {n})")
    return shards


def merge(shards: list[dict]) -> dict:
    agents = shards[0].get("agents") or []
    layers = sorted({x for s in shards for x in s.get("layers", [])})
    items: dict[str, dict] = {}
    for s in shards:
        for rec in s["items"]:
            key = rec["item_key"]
            if key in items:
                raise SystemExit(f"[merge_shards] duplicate item across "
                                 f"shards: {key} (shard {s['shard']})")
            items[key] = rec

    per_agent_dims: dict[str, dict[str, list[float]]] = {a: {} for a in agents}
    per_agent_scores: dict[str, list[float]] = {a: [] for a in agents}
    per_layer: dict[str, dict] = {}
    for rec in items.values():
        layer = rec["layer"]
        bucket = per_layer.setdefault(
            layer, {"n_items": 0,
                    "mean_score": {a: [] for a in agents}})
        bucket["n_items"] += 1
        for a in agents:
            score = rec["scores"].get(a)
            if score is None:
                continue
            per_agent_scores[a].append(score)
            bucket["mean_score"][a].append(score)
            per_agent_dims[a].setdefault(rec["dimension"], []).append(score)

    global_summary = {}
    for a in agents:
        sc = per_agent_scores[a]
        global_summary[a] = {
            "F": compute_f(per_agent_dims[a]),
            "mean_score": round(sum(sc) / len(sc), 4) if sc else 0.0,
            "n_items": len(sc),
        }
    for layer, bucket in per_layer.items():
        bucket["mean_score"] = {
            a: (round(sum(v) / len(v), 4) if v else 0.0)
            for a, v in bucket["mean_score"].items()}
    per_dimension = {
        a: {dim: round(sum(v) / len(v), 4)
            for dim, v in sorted(dims.items())}
        for a, dims in per_agent_dims.items()}

    ordered_items = [items[k] for k in sorted(items)]
    return {
        "schema": "fang_yuan_benchmark_merged_v1",
        "num_shards": shards[0]["num_shards"],
        "layers": layers,
        "agents": agents,
        "n_items": len(ordered_items),
        "global": global_summary,
        "per_layer": per_layer,
        "per_dimension": per_dimension,
        "items": ordered_items,
    }


def markdown(report: dict) -> str:
    agents = report["agents"]
    lines: list[str] = []
    lines.append("# Fang Yuan Deterministic Benchmark — merged report")
    lines.append("")
    lines.append(f"- shards merged: **{report['num_shards']}** "
                 f"(all present)")
    lines.append(f"- layers: **{', '.join(report['layers'])}** "
                 "(deterministic CI layers only)")
    lines.append(f"- items: **{report['n_items']}** × agents: "
                 f"**{', '.join(agents)}**")
    lines.append("")
    lines.append("## Global F score (0.20O + 0.15A + 0.15R + 0.15I "
                 "+ 0.15P + 0.10M + 0.10C)")
    lines.append("")
    lines.append("| agent | items | mean item score | F |")
    lines.append("|---|---:|---:|---:|")
    for a in agents:
        g = report["global"][a]
        lines.append(f"| {a} | {g['n_items']} | {g['mean_score']:.4f} | "
                     f"**{g['F']:.3f}** |")
    lines.append("")
    lines.append("## Per-layer breakdown (mean item score)")
    lines.append("")
    header = "| layer | items | " + " | ".join(agents) + " |"
    sep = "|---|---:|" + "---:|" * len(agents)
    lines += [header, sep]
    for layer in sorted(report["per_layer"]):
        b = report["per_layer"][layer]
        row = [f"{layer}", str(b["n_items"])]
        row += [f"{b['mean_score'].get(a, 0.0):.4f}" for a in agents]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="indir", default="results",
                    help="directory containing shard_*.json (default results)")
    ap.add_argument("--out", default="results/merged_benchmark.json")
    args = ap.parse_args()

    shards = load_shards(Path(args.indir))
    report = merge(shards)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    md = markdown(report)
    print(md)
    print(f"[merge_shards] wrote {out} "
          f"({report['n_items']} items from {report['num_shards']} shards)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
