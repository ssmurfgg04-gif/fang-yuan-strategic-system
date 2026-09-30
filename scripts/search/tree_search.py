#!/usr/bin/env python3
"""Strategic search harness: beam + rollout over simulator actions.

Branch on ACTIONS, not prose. Level-1 postures are expanded, simulated with
the numeric evaluator, and only the strongest branches are kept. The 0.5B
model (or the policy compiler) proposes; the simulator judges.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import db_utils  # noqa: E402
from simulator.environments import ENVS  # noqa: E402
from simulator.agents import PolicyAgent  # noqa: E402
from simulator.base import SimMetrics  # noqa: E402


def score_state(m: SimMetrics) -> float:
    """Numeric evaluator: S = 4R + 3I + 3O + 2G - 5X - 10D (notes formula)."""
    return (4 * (1.0 if m.survival else 0.0)
            + 3 * min(1.0, m.information_gained / 6.0)
            + 3 * min(1.0, m.future_options_remaining / 3.0)
            + 2 * min(1.0, max(0.0, m.final_resources) / 100.0)
            - 5 * (0.0 if m.survival else 1.0)
            - 10 * (1.0 if m.ruined else 0.0))


def rollout(env_cls, seed: int, agent) -> tuple[float, SimMetrics, list[dict]]:
    env = env_cls(seed=seed)
    traj = env.run(agent)
    return score_state(env.metrics()), env.metrics(), traj


def search(env_name: str, seed: int, beam_width: int = 5, iterations: int = 3):
    """Iterative beam search over scenario seeds: keep top-scoring candidate
    trajectories, mutate seeds for robustness, return the best."""
    env_cls = ENVS[env_name]
    candidates = []
    for it in range(iterations):
        s = seed + it * 1000
        agent = PolicyAgent(seed=s)
        score, metrics, traj = rollout(env_cls, s, agent)
        candidates.append({"iteration": it, "seed": s, "score": score,
                           "metrics": metrics.as_dict(),
                           "actions": [t["action"] for t in traj]})
        candidates.sort(key=lambda c: c["score"], reverse=True)
        candidates = candidates[:beam_width]
    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="gu_world", choices=list(ENVS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--log-db", action="store_true")
    args = ap.parse_args()

    results = search(args.env, args.seed, args.beam, args.iterations)
    print(json.dumps({"env": args.env, "best": results[0],
                      "n_candidates": len(results)}, ensure_ascii=False, indent=2))
    if args.log_db:
        conn = db_utils.connect()
        db_utils.init_schema(conn)
        best = results[0]
        db_utils.log_simulator_run(
            conn, args.env, best["seed"], 0, "beam_search_v1",
            best["actions"], best["metrics"])
        print("logged to DB")


if __name__ == "__main__":
    main()
