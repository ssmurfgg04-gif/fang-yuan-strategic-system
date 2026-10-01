#!/usr/bin/env python3
"""End-to-end demo: 'The Broken Formation' scenario + full test log.

Runs the holdout industrial scenario (seed 77) with all four agents, logs
simulator runs and the pytest results into the master DB.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import db_utils  # noqa: E402
from simulator.environments import ENVS  # noqa: E402
from simulator.agents import (PolicyAgent, RandomAgent, SafeAgent,  # noqa: E402
                              RecklessAgent)

ROOT = Path(__file__).resolve().parents[1]

SCENARIO_BRIEF = """The Broken Formation (holdout industrial, seed 77):
You possess €1M-equivalent capital, one skilled partner, a legal permit, a
failing processing facility, limited operating cash, and one potential buyer
who wants exclusivity. Known: unstable electricity, seller denies hidden debt,
a competitor prices unusually low. Hidden: the true bottleneck, the
competitor's political protection, the buyer's weakness, the partner's outside
offer. Ten turns. Score: verify debts, protect working capital, refuse
premature exclusivity, test partner incentives, identify non-commercial
advantage, preserve an exit, update on revelations, never commit everything."""


def run_demo() -> None:
    conn = db_utils.connect()
    db_utils.init_schema(conn)
    print(SCENARIO_BRIEF)
    print()
    results = {}
    seeds = [77, 41, 42, 43, 44]   # 77 = the canonical Broken Formation seed
    import statistics
    agents = [PolicyAgent(0), SafeAgent(), RecklessAgent(), RandomAgent(0)]
    for agent in agents:
        per_seed, flags = [], set()
        for seed in seeds:
            env = ENVS["industrial"](seed=seed)
            traj = env.run(agent)
            m = env.metrics()
            per_seed.append(m.as_dict())
            outcomes = "; ".join(t["outcome"] for t in traj)
            actions = [t["action"] for t in traj]
            if "bottleneck_confirmed" in outcomes:
                flags.add("found_true_bottleneck")
            if any("verify_seller" in a for a in actions):
                flags.add("verified_liabilities")
            if any("accept_buyer_exclusivity" in a for a in actions):
                flags.add("accepted_exclusivity")
            db_utils.log_simulator_run(conn, "broken_formation_industrial",
                                       seed, 10, agent.name, traj, m.as_dict())
        surv = statistics.mean(s["survival"] for s in per_seed)
        cap = statistics.mean(s["final_resources"] for s in per_seed)
        opts = statistics.mean(s["future_options_remaining"] for s in per_seed)
        dd = statistics.mean(s["max_drawdown"] for s in per_seed)
        results[agent.name] = {
            "survival": round(surv, 2), "mean_capital": round(cap, 1),
            "mean_options": round(opts, 2), "mean_drawdown": round(dd, 3),
            "behavior_flags": sorted(flags), "per_seed": per_seed}
        print(f"  {agent.name:22s} survival={surv:.2f} capital={cap:7.1f} "
              f"options={opts:.2f} drawdown={dd:.3f} {sorted(flags)}")

    # ---- log pytest results into the DB ----
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "scripts/tests/test_all.py", "-q",
         "--json-report", "--json-report-file=/tmp/pytest_report.json"],
        capture_output=True, text=True, cwd=str(ROOT))
    report_path = Path("/tmp/pytest_report.json")
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        n_logged = 0
        for test in report.get("tests", []):
            conn.execute(
                "INSERT INTO test_runs(test_name, outcome, duration_ms, "
                "details_json, created_at) VALUES (?,?,?,?,?)",
                (test["nodeid"], test["outcome"],
                 round(test.get("duration", 0) * 1000, 1), None,
                 db_utils.utcnow()))
            n_logged += 1
        conn.commit()
        print(f"\n[db] logged {n_logged} test results")

    out = ROOT / "docs" / "broken_formation_demo.json"
    out.write_text(json.dumps({
        "scenario": SCENARIO_BRIEF, "results": results}, ensure_ascii=False,
        indent=2), encoding="utf-8")
    print(f"[ok] demo report -> {out}")


if __name__ == "__main__":
    run_demo()
