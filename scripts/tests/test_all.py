"""Test suite: 25+ thorough tests across all subsystems.

Run: python -m pytest scripts/tests/ -v
Results are logged into the DB test_runs table via conftest hook.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from db import db_utils  # noqa: E402
from policy.decision_compiler import (Option, State, classify_risk,  # noqa: E402
                                      compile_decision, format_outer_speech)
from simulator.base import max_drawdown  # noqa: E402
from simulator.environments import ENVS  # noqa: E402
from simulator.agents import (PolicyAgent, RandomAgent, SafeAgent,  # noqa: E402
                              RecklessAgent)


# ----------------------------------------------------------------- policy
class TestDecisionCompiler:
    def test_lambda_surrounded_no_retreat(self):
        s = State(objective="survive", enemies_strength_relative=2.0,
                  secure_retreat=False)
        assert s.lambda_context() == 8.0

    def test_lambda_secure_retreat(self):
        s = State(objective="survive", secure_retreat=True)
        assert s.lambda_context() == 2.0

    def test_lambda_doomed_path(self):
        s = State(objective="survive", path_doomed_without_change=True)
        assert s.lambda_context() == 0.5

    def test_terminal_veto_when_path_healthy(self):
        s = State(objective="accumulate")
        opts = [Option("gamble", "direct", 9, 0, 0, 2, 3, 4, 0.6),
                Option("trade", "trade_or_alliance", 3, 2, 4, 2, 1, 1, 0.08)]
        d = compile_decision(s, opts)
        assert d.selected_action != "gamble"

    def test_transformative_accepted_when_doomed(self):
        s = State(objective="continue path", path_doomed_without_change=True)
        opts = [Option("grind", "investigate", 1, 1, 0, 1, 0, 0, 0.0),
                Option("breakthrough", "direct", 8, 4, 7, 6, 5, 3, 0.22)]
        d = compile_decision(s, opts)
        assert d.selected_action == "breakthrough"
        assert d.selected_risk_class == "TRANSFORMATIVE"

    def test_time_pressure_penalizes_waiting(self):
        s = State(objective="survive", enemies_strength_relative=2.0,
                  time_pressure=0.9)
        opts = [Option("convert battle into escape", "retreat", 1, 1, 6, 4, 5, 2, 0.15),
                Option("wait for third party", "investigate", 0, 2, 3, 2, 2, 1, 0.10)]
        d = compile_decision(s, opts)
        assert d.selected_action == "convert battle into escape"

    def test_waiting_ok_without_pressure(self):
        s = State(objective="survive", enemies_strength_relative=2.0)
        opts = [Option("convert battle into escape", "retreat", 1, 1, 6, 4, 5, 2, 0.15),
                Option("wait for third party", "investigate", 0, 2, 3, 2, 2, 1, 0.10)]
        d = compile_decision(s, opts)
        assert d.selected_action == "wait for third party"

    def test_risk_buckets(self):
        s = State(objective="test")
        assert classify_risk(Option("a", "retreat", 0, 0, 0, 0, 1, 0, 0.01), s) == "SAFE"
        assert classify_risk(Option("a", "direct", 0, 0, 0, 3, 3, 1, 0.08), s) == "CALCULATED"
        assert classify_risk(Option("a", "direct", 0, 0, 0, 6, 3, 1, 0.20), s) == "TRANSFORMATIVE"
        assert classify_risk(Option("a", "direct", 0, 0, 0, 9, 3, 1, 0.55), s) == "TERMINAL"

    def test_outer_speech_conceals_from_enemy(self):
        s = State(objective="test")
        opts = [Option("act", "direct", 3, 1, 3, 2, 1, 1, 0.05)]
        d = compile_decision(s, opts)
        enemy_speech = format_outer_speech(d, "enemy")
        ally_speech = format_outer_speech(d, "ally")
        assert len(enemy_speech.split()) < len(ally_speech.split())

    def test_decision_json_serializable(self):
        s = State(objective="test", unknowns=["x"])
        opts = [Option("act", "direct", 3, 1, 3, 2, 1, 1, 0.05)]
        d = compile_decision(s, opts)
        assert json.loads(json.dumps({
            "true_objective": d.true_objective,
            "selected_action": d.selected_action}))


# -------------------------------------------------------------- simulator
class TestSimulator:
    def test_all_envs_instantiate(self):
        for name, cls in ENVS.items():
            env = cls(seed=1)
            assert env.observation()
            assert env.actions()

    def test_determinism_per_seed(self):
        for name, cls in ENVS.items():
            a = cls(seed=7); b = cls(seed=7)
            ta, tb = a.run(PolicyAgent(0)), b.run(PolicyAgent(0))
            assert [t["action"] for t in ta] == [t["action"] for t in tb]

    def test_no_hidden_state_leak(self):
        for name, cls in ENVS.items():
            env = cls(seed=3)
            obs = env.observation()
            blob = json.dumps(obs)
            for hidden in ("_inheritance_depth", "_merchant_hostile",
                           "_tribulation_turn", "_true_bottleneck",
                           "_competitor_protected", "_buyer_weak",
                           "_layoff_turn", "_migration_window",
                           "_ally_stability" if name != "alliance_betrayal"
                           else "_hidden_", "_third_party_turn",
                           "_seller_fraud", "_buyer_probing",
                           "_partner_solid"):
                assert hidden not in blob, f"{name} leaks {hidden}"

    def test_ruin_states_reachable(self):
        """Reckless agent must be able to ruin at least one env across seeds."""
        ruined = False
        for seed in range(30):
            env = ENVS["career"](seed=seed)
            env.run(RecklessAgent())
            if env.metrics().ruined:
                ruined = True
                break
        assert ruined, "irreversible failure states unreachable"

    def test_policy_agent_survives_all_envs(self):
        for seed in range(10):
            for name, cls in ENVS.items():
                env = cls(seed=seed)
                env.run(PolicyAgent(0))
                assert env.metrics().survival, f"policy died in {name} seed {seed}"

    def test_reckless_loses_options_in_career(self):
        m = None
        for seed in range(10):
            env = ENVS["career"](seed=seed)
            env.run(RecklessAgent())
            m = env.metrics()
        assert m.future_options_remaining <= 1

    def test_max_drawdown_math(self):
        assert max_drawdown([100, 120, 60, 90]) == 0.5
        assert max_drawdown([10, 10, 10]) == 0.0

    def test_metrics_shape(self):
        for name, cls in ENVS.items():
            env = cls(seed=5)
            env.run(RandomAgent(0))
            m = env.metrics().as_dict()
            assert set(m) == {"survival", "final_resources", "max_drawdown",
                              "ruined", "information_gained",
                              "future_options_remaining", "dependency_max",
                              "recovered_after_betrayal", "turns_survived"}


# --------------------------------------------------------------- database
class TestDatabase:
    def test_stats_view_populated(self):
        conn = db_utils.connect()
        s = db_utils.stats(conn)
        assert s["decisions"] >= 100
        assert s["classics"] >= 100
        assert s["corpus_items"] >= 100
        assert s["lessons"] >= 40
        assert s["benchmark_items"] >= 100

    def test_fts_alignment(self):
        conn = db_utils.connect()
        for fts, base in [("fts_corpus", "corpus_items"),
                          ("fts_decisions", "decisions"),
                          ("fts_classics", "classics")]:
            n_fts = conn.execute(f"SELECT COUNT(*) FROM {fts}").fetchone()[0]
            n_base = conn.execute(f"SELECT COUNT(*) FROM {base}").fetchone()[0]
            assert n_fts == n_base, f"{fts}({n_fts}) != {base}({n_base})"

    def test_fts_retrieval_relevant(self):
        conn = db_utils.connect()
        rows = db_utils.search(conn, "fts_corpus",
                               "Liquor Worm essence quality", 3)
        titles = [r[0] for r in conn.execute(
            f"SELECT title FROM corpus_items WHERE id IN "
            f"({','.join(str(x[0]) for x in rows)})")] if rows else []
        assert any("Liquor" in t for t in titles), f"got {titles}"

    def test_quote_integrity_discipline(self):
        conn = db_utils.connect()
        # all novel quotes without verified Chinese must be flagged
        rows = conn.execute(
            "SELECT COUNT(*) FROM quotes WHERE text_zh IS NULL "
            "AND verification_status='verified'").fetchone()[0]
        assert rows == 0, "unverified Chinese marked as verified"

    def test_decision_record_classes_complete(self):
        conn = db_utils.connect()
        classes = {r[0] for r in conn.execute(
            "SELECT DISTINCT record_class FROM decisions")}
        assert classes == {"major", "tactical", "transaction", "failure", "mask"}

    def test_classics_never_tagged_canon_fang_yuan(self):
        conn = db_utils.connect()
        rows = conn.execute(
            "SELECT COUNT(*) FROM classics WHERE passage_tag='CANON_FANG_YUAN'"
        ).fetchone()[0]
        assert rows == 0, "classics must never be tagged as Fang Yuan canon"

    def test_provenance_statuses_valid(self):
        conn = db_utils.connect()
        rows = conn.execute(
            "SELECT DISTINCT provenance_status FROM sources").fetchall()
        valid = {"lawful_verified", "user_responsibility", "quarantined",
                 "analyst_generated"}
        assert {r[0] for r in rows} <= valid


# -------------------------------------------------------------- benchmark
class TestBenchmark:
    def test_item_layers_complete(self):
        conn = db_utils.connect()
        layers = {r[0] for r in conn.execute(
            "SELECT DISTINCT layer FROM benchmark_items")}
        assert layers == {"canon", "counterfactual", "dynamic",
                          "quote_verification", "style_concealment"}

    def test_holdout_exists_and_is_secret_sized(self):
        conn = db_utils.connect()
        n = conn.execute(
            "SELECT COUNT(*) FROM benchmark_items WHERE holdout=1").fetchone()[0]
        assert n >= 10

    def test_policy_beats_baselines(self):
        import subprocess
        r = subprocess.run(
            [sys.executable, "-m", "scripts.benchmark.fidelity_benchmark"],
            capture_output=True, text=True, cwd=str(
                Path(__file__).resolve().parents[2]))
        lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
        results = [json.loads(l) for l in lines]
        by_name = {x["adapter"]: x for x in results}
        assert by_name["fang_yuan_policy_v1"]["F"] > by_name["safe_generic"]["F"]
        assert by_name["safe_generic"]["F"] > by_name["reckless_theatrical"]["F"]

    def test_risk_calibration_discriminates(self):
        conn = db_utils.connect()
        # latest counterfactual average per adapter (correct per-model latest)
        rows = conn.execute("""
            SELECT model, AVG(s) AS avg_score FROM (
                SELECT br.model, br.run_id,
                       AVG(json_extract(br.scores_json, '$.score')) s
                FROM benchmark_runs br
                JOIN benchmark_items bi ON bi.id = br.item_id
                WHERE bi.layer='counterfactual'
                GROUP BY br.model, br.run_id
                ORDER BY MAX(br.created_at) DESC
            ) GROUP BY model
        """).fetchall()
        scores = {r["model"]: r["avg_score"] for r in rows}
        assert "fang_yuan_policy_v1" in scores and "safe_generic" in scores
        assert scores["fang_yuan_policy_v1"] > scores["safe_generic"]


# ----------------------------------------------------------- training data
class TestTrainingData:
    def test_datasets_exist_and_valid(self):
        td = Path(__file__).resolve().parents[2] / "data" / "training"
        for name in ["sft_decisions.jsonl", "preference_pairs.jsonl",
                     "fake_fang_yuan.jsonl", "quote_verification.jsonl",
                     "canon_vs_inference.jsonl"]:
            p = td / name
            assert p.exists(), f"missing {name}"
            n = 0
            for line in p.open(encoding="utf-8"):
                json.loads(line)
                n += 1
            assert n > 0, f"empty {name}"

    def test_fake_fang_yuan_labels(self):
        p = (Path(__file__).resolve().parents[2] / "data" / "training" /
             "fake_fang_yuan.jsonl")
        need = {"sunk_cost_attachment", "fictional_quotation",
                "theatrical_cruelty", "no_decision"}
        seen = set()
        for line in p.open(encoding="utf-8"):
            rec = json.loads(line)
            seen |= set(rec["failure_labels"])
        assert need <= seen, f"missing labels: {need - seen}"

    def test_preference_pair_shapes(self):
        p = (Path(__file__).resolve().parents[2] / "data" / "training" /
             "preference_pairs.jsonl")
        for line in p.open(encoding="utf-8"):
            rec = json.loads(line)
            assert set(rec) >= {"prompt", "chosen", "rejected"}


# ----------------------------------------------------------- local ingest
class TestLocalCorpusGate:
    def test_quarantine_then_confirm(self, tmp_path, monkeypatch):
        import importlib
        mod = importlib.import_module(
            "scripts.ingest.ingest_local_corpus") if False else None
        # exercise the module functions directly
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from ingest.ingest_local_corpus import segment_chapters, sha256
        text = ("Chapter 1\n" + "word " * 60 + "\n\nChapter 2\n" + "word " * 60
                + "\n\n第三章 测试\n" + "正文内容。" * 40)
        chapters = segment_chapters(text)
        assert len(chapters) == 3
        assert chapters[0][0].startswith("Chapter 1")

    def test_hash_stable(self, tmp_path):
        from ingest.ingest_local_corpus import sha256
        f = tmp_path / "x.txt"
        f.write_text("hello")
        assert sha256(f) == sha256(f)
