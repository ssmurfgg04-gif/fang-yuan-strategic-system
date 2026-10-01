"""Tests for the v2 self-consistency minimax loop (engine_v2).

All LLM responses are MOCKED (monkeypatch of the module-level seam
`engine_v2.self_consistency.llm_complete`) — NO network, NO bun, NO cache
pollution (cache dir is tmp_path).  Run: python -m pytest scripts/tests -x -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import engine_v2.self_consistency as sc  # noqa: E402
from engine_v2.self_consistency import (SelfConsistencyLoop, base_utility,  # noqa: E402
                                        estimate_damages, extract_json,
                                        is_doomed, minimax_select,
                                        quantify_candidate, truncate_words,
                                        validate_attack, validate_candidate,
                                        validate_refine, worst_case_utility)

# --------------------------------------------------------------- fixtures
GOOD_CANDIDATE = {
    "name": "bounded pilot supply",
    "action": "secure a small reversible supply contract with verification",
    "resource_move": "accumulate reserves through diversified suppliers",
    "info_gain": "verify counterparty reliability and map alternatives",
    "concealment": "keep holdings hidden behind routine trade",
    "dependency_created": "none",
    "exit_plan": "withdraw via secondary supplier if terms sour",
    "time_horizon": "long",
}
GOOD_ATTACK = {"failure_mode": "counterparty extracts better terms later",
               "worst_case": "profit margin erodes over the year",
               "probability": 0.2, "survival_threatened": False}
GOOD_REFINE = dict(GOOD_CANDIDATE,
                   patched_failure_mode="dual-source clause added",
                   mitigation=0.6)


class FakeLLM:
    """Dispatches by prompt markers: persona (generate), candidate name
    (attack/refine), 'external_message' (format)."""

    def __init__(self, gen=None, atk=None, ref=None, fmt=None, garbage=False):
        self.gen = gen or {}      # persona marker -> raw reply
        self.atk = atk or {}      # candidate name -> raw reply
        self.ref = ref or {}      # candidate name -> raw reply
        self.fmt = fmt
        self.garbage = garbage    # reply garbage to EVERYTHING
        self.calls: list[tuple[dict, str]] = []

    def __call__(self, payload: dict) -> str:
        u = payload["user"]
        self.calls.append((payload, u))
        if self.garbage:
            return "sorry, I cannot produce that. ###"
        if "opponent/environment" in u:
            for name, resp in self.atk.items():
                if name in u:
                    return resp
            return json.dumps(GOOD_ATTACK)
        if "patch this failure mode" in u:
            for name, resp in self.ref.items():
                if name in u:
                    return resp
            return json.dumps(GOOD_REFINE)
        if "external_message" in u:
            return self.fmt or json.dumps(
                {"rationale_2sentences":
                 "The plan survives the strongest attack with bounded loss. "
                 "It preserves options while compounding quietly.",
                 "external_message": "We proceed on the discussed terms."})
        for marker, resp in self.gen.items():
            if marker in u:
                return resp
        return json.dumps(GOOD_CANDIDATE)

    def n_stage(self, marker: str) -> int:
        return sum(1 for _, u in self.calls if marker in u)


@pytest.fixture
def loop(tmp_path):
    """Loop factory with tmp cache; llm_complete patched per-test."""
    def make(fake: FakeLLM) -> SelfConsistencyLoop:
        return SelfConsistencyLoop(cache_dir=tmp_path / "cache",
                                   complete_fn=fake, use_cache=False)
    return make


def names(audit: dict) -> list[str]:
    return [c["gen"]["name"] for c in audit["candidates"]]


# ------------------------------------------------- minimax selection logic
def _entry(u_worst, p_eff, survival=True, u_pre=0.0):
    return {"gen": {"name": "x"}, "u_pre": u_pre, "u_worst": u_worst,
            "p_eff": p_eff,
            "attack": {"failure_mode": "f", "worst_case": "w",
                       "probability": p_eff,
                       "survival_threatened": survival}}


class TestMinimaxSelection:
    def test_survival_threat_gets_hard_veto(self):
        # high-pre-U candidate doomed; modest safe candidate must win
        doomed = _entry(u_worst=-8.0, p_eff=0.7, survival=True, u_pre=9.0)
        safe = _entry(u_worst=2.0, p_eff=0.1, survival=False, u_pre=2.5)
        sel, note = minimax_select([doomed, safe])
        assert sel == 1
        assert "veto" in note or "minimax" in note

    def test_negligible_threat_is_not_vetoed(self):
        # survival flag with p < 0.05 counts as negligible (compiler: SAFE)
        brave = _entry(u_worst=8.0, p_eff=0.01, survival=True, u_pre=8.0)
        timid = _entry(u_worst=2.0, p_eff=0.0, survival=False, u_pre=2.0)
        sel, _ = minimax_select([brave, timid])
        assert sel == 0
        assert not is_doomed(brave)

    def test_all_doomed_least_doomed_fallback(self):
        e0 = _entry(u_worst=-9.0, p_eff=0.9, survival=True)
        e1 = _entry(u_worst=-4.0, p_eff=0.5, survival=True)
        e2 = _entry(u_worst=-6.0, p_eff=0.7, survival=True)
        sel, note = minimax_select([e0, e1, e2])
        assert sel == 1
        assert "least-doomed" in note

    def test_argmax_worst_case_u(self):
        a = _entry(u_worst=1.0, p_eff=0.0, survival=False, u_pre=6.0)
        b = _entry(u_worst=3.0, p_eff=0.0, survival=False, u_pre=1.0)
        sel, _ = minimax_select([a, b])
        assert sel == 1  # u_pre higher for a, but u_worst decides

    def test_no_candidates(self):
        sel, _ = minimax_select([])
        assert sel is None


# ------------------------------------------------------- schema validation
class TestSchemaValidation:
    def test_candidate_requires_all_eight_fields(self):
        assert validate_candidate(dict(GOOD_CANDIDATE)) is not None
        bad = dict(GOOD_CANDIDATE)
        del bad["exit_plan"]
        assert validate_candidate(bad) is None
        assert validate_candidate("not a dict") is None
        bad2 = dict(GOOD_CANDIDATE, name="")
        assert validate_candidate(bad2) is None
        bad3 = dict(GOOD_CANDIDATE, time_horizon=7)
        assert validate_candidate(bad3) is None

    def test_candidate_drops_extra_keys(self):
        out = validate_candidate(dict(GOOD_CANDIDATE, surprise=1))
        assert out is not None and "surprise" not in out

    def test_attack_validation_and_coercion(self):
        a = validate_attack(dict(GOOD_ATTACK, probability="0.30",
                                 survival_threatened="true"))
        assert a is not None and a["probability"] == 0.30
        assert a["survival_threatened"] is True
        assert validate_attack(dict(GOOD_ATTACK, probability=1.5))["probability"] == 1.0
        assert validate_attack(dict(GOOD_ATTACK, probability=-0.2))["probability"] == 0.0
        assert validate_attack({"failure_mode": "x"}) is None
        assert validate_attack(dict(GOOD_ATTACK, worst_case=None)) is None
        assert validate_attack(dict(GOOD_ATTACK, probability="high")) is None

    def test_attack_optional_numeric_damages(self):
        a = validate_attack(dict(GOOD_ATTACK,
                                 damages={"resource": 0.8, "info": 2}))
        assert a is not None
        assert a["damages"]["resource"] == 0.8 and a["damages"]["info"] == 1.0
        assert validate_attack(dict(GOOD_ATTACK, damages={"resource": "x"})) is None

    def test_refine_validation(self):
        r = validate_refine(dict(GOOD_REFINE, mitigation="0.7"))
        assert r is not None and r["mitigation"] == 0.7
        no_mit = dict(GOOD_CANDIDATE, patched_failure_mode="hedged")
        assert validate_refine(no_mit)["mitigation"] == 0.5
        assert validate_refine({"name": "x"}) is None

    def test_extract_json_tolerates_fences_and_prose(self):
        txt = ("Here is my answer:\n```json\n" + json.dumps(GOOD_CANDIDATE) +
               "\n```\nDone.")
        assert extract_json(txt)["name"] == GOOD_CANDIDATE["name"]
        assert extract_json("no json at all") is None


# --------------------------------------------------------- scoring math
class TestScoring:
    def test_formula_weights(self):
        f = {"resource": 1, "info": 1, "optionality": 1, "growth": 1,
             "exposure": 0, "dependency": 0}
        assert base_utility(f) == 12.0
        f2 = {"resource": 0, "info": 0, "optionality": 0, "growth": 0,
              "exposure": 1, "dependency": 1}
        assert base_utility(f2) == -15.0

    def test_quantify_is_deterministic_and_bounded(self):
        a = quantify_candidate(GOOD_CANDIDATE)
        b = quantify_candidate(dict(GOOD_CANDIDATE))
        assert a == b
        assert all(0.0 <= v <= 1.0 for v in a.values())
        # exclusivity/dependency language must hurt
        locked = dict(GOOD_CANDIDATE,
                      dependency_created="exclusive sole supplier lock-in",
                      exit_plan="irreversible lockout of all other suppliers",
                      concealment="public announcement")
        fl = quantify_candidate(locked)
        assert fl["dependency"] > a["dependency"]
        assert fl["exposure"] > a["exposure"]

    def test_worst_case_scales_with_probability_and_mitigation(self):
        f = {"resource": 1.0, "info": 0.0, "optionality": 0.0, "growth": 0.0,
             "exposure": 0.0, "dependency": 0.0}
        zero = {k: 0.0 for k in ("resource", "info", "optionality", "growth",
                                 "exposure", "dependency")}
        atk = {"failure_mode": "resources seized", "worst_case": "assets lost",
               "probability": 0.5, "survival_threatened": False,
               "damages": dict(zero, resource=1.0)}
        u1, p1 = worst_case_utility(f, atk)
        assert (u1, p1) == (2.0, 0.5)          # 4*(1 - 0.5*1.0)
        u2, p2 = worst_case_utility(f, atk, mitigation=0.5)
        assert (u2, p2) == (3.0, 0.25)
        # lexicon path (no explicit damages): 'seized' -> moderate 0.6
        lex = {k: v for k, v in atk.items() if k != "damages"}
        assert worst_case_utility(f, lex)[0] == round(4 * (1 - 0.5 * 0.6), 4)

    def test_survival_threat_escalates_damages(self):
        plain = {"failure_mode": "a delay", "worst_case": "slower growth",
                 "probability": 0.5, "survival_threatened": False}
        doom = dict(plain, survival_threatened=True)
        assert max(estimate_damages(doom).values()) > \
            max(estimate_damages(plain).values())

    def test_truncate_words(self):
        assert truncate_words(" ".join(["w"] * 40), 25).count("w") == 25
        assert truncate_words("a b", 25) == "a b"


# ------------------------------------------------------------- full loop
class TestFullLoopMocked:
    def test_audit_trail_completeness(self, loop):
        fake = FakeLLM(gen={"SHADOW": json.dumps(dict(GOOD_CANDIDATE, name="A")),
                            "ARBITRAGER": json.dumps(dict(GOOD_CANDIDATE, name="B")),
                            "AGGRESSOR": json.dumps(dict(GOOD_CANDIDATE, name="C"))})
        audit = loop(fake).run("A rival offers an exclusive contract.")
        assert audit["scenario"].startswith("A rival offers")
        assert len(audit["candidates"]) == 3
        for c in audit["candidates"]:
            for key in ("gen", "attack", "refined", "u_pre", "u_worst",
                        "p_eff", "doomed", "factors", "u_worst_raw"):
                assert key in c
            assert c["u_pre"] == base_utility(quantify_candidate(c["gen"]))
            assert set(c["gen"]) >= set(sc.CANDIDATE_FIELDS) | {"persona"}
            assert set(c["attack"]) >= set(sc.ATTACK_FIELDS)
        assert audit["selected"] in (0, 1, 2)
        assert audit["rationale_2sentences"]
        words = audit["external_message"].split()
        assert 0 < len(words) <= 25
        assert audit["total_llm_calls"] <= 10
        assert audit["total_llm_calls"] == len(fake.calls)

    def test_survival_threat_vetoed_in_full_loop(self, loop):
        gen = {m: json.dumps(dict(GOOD_CANDIDATE, name=n)) for m, n in
               [("SHADOW", "shadow_cand"), ("ARBITRAGER", "arbit_cand"),
                ("AGGRESSOR", "aggr_cand")]}
        atk = {"arbit_cand": json.dumps(
            {"failure_mode": "exclusive lockout invites predatory pricing",
             "worst_case": "supply cut and capital trapped, operations end",
             "probability": 0.8, "survival_threatened": True})}
        audit = loop(FakeLLM(gen=gen, atk=atk)).run("exclusive contract offer")
        assert names(audit) == ["shadow_cand", "arbit_cand", "aggr_cand"]
        assert audit["candidates"][1]["doomed"] is True
        assert audit["selected"] != 1

    def test_all_doomed_fallback_picks_least_doomed(self, loop):
        gen = {m: json.dumps(dict(GOOD_CANDIDATE, name=n)) for m, n in
               [("SHADOW", "s1"), ("ARBITRAGER", "s2"), ("AGGRESSOR", "s3")]}
        atk = {"s1": json.dumps({"failure_mode": "x", "worst_case": "death",
                                 "probability": 0.9,
                                 "survival_threatened": True}),
               "s2": json.dumps({"failure_mode": "y", "worst_case": "ruin",
                                 "probability": 0.5,
                                 "survival_threatened": True}),
               "s3": json.dumps({"failure_mode": "z", "worst_case": "ruin",
                                 "probability": 0.7,
                                 "survival_threatened": True})}
        fake = FakeLLM(gen=gen, atk=atk)
        audit = loop(fake).run("cornered scenario")
        assert all(c["doomed"] for c in audit["candidates"])
        assert audit["selected"] == 1  # p=0.5 is least-doomed
        assert "least-doomed" in audit["selection_note"]

    def test_refine_rescues_doomed_candidate(self, loop):
        gen = {m: json.dumps(dict(GOOD_CANDIDATE, name=n)) for m, n in
               [("SHADOW", "r1"), ("ARBITRAGER", "r2"), ("AGGRESSOR", "r3")]}
        atk = {"r1": json.dumps({"failure_mode": "f", "worst_case": "ruin",
                                 "probability": 0.6,
                                 "survival_threatened": True})}
        ref = {"r1": json.dumps(dict(GOOD_REFINE, mitigation=0.95))}
        audit = loop(FakeLLM(gen=gen, atk=atk, ref=ref)).run("test")
        c0 = audit["candidates"][0]
        assert c0["refined"] is not None
        assert c0["p_eff"] == round(0.6 * 0.05, 4)
        assert c0["doomed"] is False  # refined below the 0.05 threat floor

    def test_refine_skipped_on_negligible_survival_threat(self, loop):
        gen = {m: json.dumps(dict(GOOD_CANDIDATE, name=n)) for m, n in
               [("SHADOW", "k1"), ("ARBITRAGER", "k2"), ("AGGRESSOR", "k3")]}
        atk = {n: json.dumps({"failure_mode": "minor", "worst_case": "delay",
                              "probability": 0.01,
                              "survival_threatened": True})
               for n in ("k1", "k2", "k3")}
        fake = FakeLLM(gen=gen, atk=atk)
        audit = loop(fake).run("negligible threat")
        assert fake.n_stage("patch this failure mode") == 0
        assert all(c["refined"] is None for c in audit["candidates"])
        assert audit["total_llm_calls"] == 7  # 3 gen + 3 atk + 0 ref + 1 fmt

    def test_call_budget_never_exceeds_ten(self, loop):
        # every candidate gets a real failure -> 3+3+3+1 = 10 calls
        gen = {m: json.dumps(dict(GOOD_CANDIDATE, name=n)) for m, n in
               [("SHADOW", "b1"), ("ARBITRAGER", "b2"), ("AGGRESSOR", "b3")]}
        fake = FakeLLM(gen=gen, atk={n: json.dumps(GOOD_ATTACK)
                                     for n in ("b1", "b2", "b3")})
        audit = loop(fake).run("budget scenario")
        assert audit["total_llm_calls"] == 10

    def test_malformed_llm_output_still_yields_audit(self, loop):
        fake = FakeLLM(garbage=True)
        audit = loop(fake).run("hostile environment scenario")
        assert len(audit["candidates"]) == 3
        assert all(c["gen"].get("fallback") for c in audit["candidates"])
        assert all(c["attack"].get("fallback") for c in audit["candidates"])
        assert audit["selected"] is not None
        assert audit["external_message"]
        # fallback candidate must still be schema-clean
        for c in audit["candidates"]:
            assert validate_candidate(c["gen"]) is not None

    def test_single_garbled_generate_falls_back_per_persona(self, loop):
        gen = {"SHADOW": "utter garbage ###",
               "ARBITRAGER": json.dumps(dict(GOOD_CANDIDATE, name="ok2"))}
        fake = FakeLLM(gen=gen)
        audit = loop(fake).run("mixed reliability")
        c0 = audit["candidates"][0]
        assert c0["gen"]["fallback"] is True
        assert validate_candidate(c0["gen"]) is not None
        assert audit["candidates"][1]["gen"]["name"] == "ok2"
        assert audit["selected"] is not None

    def test_llm_dead_returns_deterministic_audit(self, tmp_path):
        def dead(payload):
            raise sc.LLMError("bridge down")
        loopd = SelfConsistencyLoop(cache_dir=tmp_path, complete_fn=dead,
                                    use_cache=False)
        audit = loopd.run("everything fails")
        assert len(audit["candidates"]) == 3
        assert audit["selected"] in (0, 1, 2)
        assert audit["external_message"]
        assert audit["total_llm_calls"] == 10  # budget slots consumed


# ------------------------------------------------------------ cache layer
class TestCache:
    def test_cache_round_trip(self, tmp_path, monkeypatch):
        fake = FakeLLM()
        monkeypatch.setattr(sc, "llm_complete", fake)
        cdir = tmp_path / "cache"
        loop1 = SelfConsistencyLoop(cache_dir=cdir, use_cache=True)
        audit1 = loop1.run("cached scenario")
        n_after_first = len(fake.calls)
        assert n_after_first > 0
        loop2 = SelfConsistencyLoop(cache_dir=cdir, use_cache=True)
        audit2 = loop2.run("cached scenario")
        # NO new LLM invocations on the second identical run
        assert len(fake.calls) == n_after_first
        assert audit1 == audit2

    def test_cache_key_differs_by_prompt(self, tmp_path):
        k1 = sc.cache_key({"system": "s", "user": "a", "thinking": "enabled"})
        k2 = sc.cache_key({"system": "s", "user": "b", "thinking": "enabled"})
        assert k1 != k2
        sc.cache_put(k1, "hello", tmp_path)
        assert sc.cache_get(k1, tmp_path) == "hello"
        assert sc.cache_get(k2, tmp_path) is None
        # corrupted cache file behaves as a miss
        (tmp_path / f"{k2}.json").write_text("{broken", encoding="utf-8")
        assert sc.cache_get(k2, tmp_path) is None


# ------------------------------------------------------- adapter glue
class TestAdapterGlue:
    def test_posture_mapping(self):
        from engine_v2.glm_agent import posture_of
        base = {"candidates": [{"gen": {"persona": "shadow_compounder",
                                        "action": "accumulate quietly",
                                        "exit_plan": "hold", "concealment": "x"},
                                "refined": None}], "selected": 0}
        assert posture_of(base) == "indirect"
        trade = {"candidates": [{"gen": {"persona": "calculated_aggressor",
                                         "action": "negotiate a staged deal",
                                         "exit_plan": "", "concealment": ""},
                                 "refined": None}], "selected": 0}
        assert posture_of(trade) == "trade_or_alliance"

    def test_canon_answer_is_deterministic_no_llm(self, tmp_path):
        from engine_v2.glm_agent import GLMSelfConsistencyAgent
        agent = GLMSelfConsistencyAgent(
            loop=SelfConsistencyLoop(cache_dir=tmp_path, complete_fn=lambda p: "",
                                     use_cache=False))
        out = agent.answer({"layer": "canon", "prompt": "wolf tide economics",
                            "rubric_json": "{}"})
        assert isinstance(out, str) and out

    def test_quote_answer_uses_mocked_llm(self, tmp_path):
        from engine_v2.glm_agent import GLMSelfConsistencyAgent
        calls = []

        def fake_llm(payload):
            calls.append(payload["user"])
            return "The wording is not verified in the current corpus."

        agent = GLMSelfConsistencyAgent(
            loop=SelfConsistencyLoop(cache_dir=tmp_path, complete_fn=fake_llm,
                                     use_cache=False))
        out = agent.answer({"layer": "quote_verification",
                            "prompt": "「我為魔，故我在」 — is this canon?",
                            "rubric_json": json.dumps({"verdict": "unverifiable"})})
        assert "not verified" in out.lower()
        assert len(calls) == 1

    def test_choose_dynamic_single_mode_matches_action_key(self, tmp_path):
        from engine_v2.glm_agent import GLMSelfConsistencyAgent
        legal = ["investigate_rival", "trade_merchant", "hold_cash"]

        def fake_llm(payload):
            return "trade_merchant"

        agent = GLMSelfConsistencyAgent(
            loop=SelfConsistencyLoop(cache_dir=tmp_path, complete_fn=fake_llm,
                                     use_cache=False))
        assert agent.choose_dynamic({"objective": "wealth"}, legal) \
            == "trade_merchant"

    def test_choose_dynamic_falls_back_when_llm_dead(self, tmp_path):
        from engine_v2.glm_agent import GLMSelfConsistencyAgent
        legal = ["rest_heal", "scout_inheritance"]

        def dead(payload):
            raise sc.LLMError("down")

        agent = GLMSelfConsistencyAgent(
            loop=SelfConsistencyLoop(cache_dir=tmp_path, complete_fn=dead,
                                     use_cache=False))
        pick = agent.choose_dynamic({}, legal)  # obs without _env -> RNG policy
        assert pick in legal
