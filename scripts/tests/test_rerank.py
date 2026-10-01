"""Tests for the engine_v2 two-stage retrieval + LLM cross-reranker.

ALL LLM calls are mocked (score_candidates monkeypatched) — no network.
DB access is read-only against the local SQLite file.
Run: python -m pytest scripts/tests/test_rerank.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from db import db_utils  # noqa: E402
import importlib  # noqa: E402

# NOTE: `from engine_v2 import rerank` would bind the re-exported FUNCTION
# (see engine_v2/__init__.py); importlib gets the actual submodule.
R = importlib.import_module("engine_v2.rerank")  # noqa: E402

SCENARIO_APERTURE = "aperture awakening C-grade twin brother clan attention"


@pytest.fixture()
def conn():
    c = db_utils.connect()
    yield c
    c.close()


def _ok_scores(candidates, base=7):
    """Mock scorer payload: distinct descending scores in recall order."""
    return {c["key"]: {"score": max(0, base - i) % 11, "reason": f"mock {i}",
                       "ok": True}
            for i, c in enumerate(candidates)}


# ------------------------------------------------------------------ stage 1
class TestRecall:
    def test_recall_finds_aperture_decision(self, conn):
        """Known scenario must recall decision id 1 as the top bm25 hit."""
        cands = R.recall_candidates(conn, SCENARIO_APERTURE)
        assert any(c["source"] == "decisions" and c["id"] == 1
                   for c in cands)
        d1 = next(c for c in cands
                  if c["source"] == "decisions" and c["id"] == 1)
        assert d1["bm25_rank"] == 0
        assert "aperture" in d1["text"].lower()

    def test_recall_interleaves_sources_and_caps(self, conn):
        cands = R.recall_candidates(
            conn, "exclusivity deception concealment exposure war",
            recall_size=20)
        assert len(cands) <= 20
        srcs = {c["source"] for c in cands}
        assert srcs <= {"decisions", "classics"}
        assert len(srcs) == 2, "interleaving should keep both sources"
        # deterministic: same input, same order
        again = R.recall_candidates(
            conn, "exclusivity deception concealment exposure war",
            recall_size=20)
        assert [c["key"] for c in cands] == [c["key"] for c in again]

    def test_like_fallback_when_fts_empty(self, conn, monkeypatch):
        monkeypatch.setattr(
            R.db_utils, "search",
            lambda *a, **k: [])  # simulate dead/empty FTS
        cands = R.recall_candidates(
            conn, "aperture awakening ceremony brother ridicule",
            sources=("decisions",), recall_size=5)
        assert cands, "LIKE fallback returned nothing"
        assert any(c["id"] == 1 for c in cands)
        # fallback ranks sort after any FTS hit
        assert all(c["bm25_rank"] >= 1000 for c in cands)

    def test_empty_scenario_recall_safe(self, conn):
        assert R.recall_candidates(conn, "") == []


# ------------------------------------------------------------------ stage 2
class TestScoring:
    def test_parse_score_response_variants(self):
        assert R.parse_score_response('{"score": 8, "reason": "good"}') == \
            {"score": 8, "reason": "good"}
        fenced = "```json\n{\"score\": 9, \"reason\": \"matches\"}\n```"
        assert R.parse_score_response(fenced)["score"] == 9
        assert R.parse_score_response('{"score": "7", "reason": "s"}')\
            ["score"] == 7
        assert R.parse_score_response('{"score": 14, "reason": "hi"}')\
            ["score"] == 10
        assert R.parse_score_response('{"score": -3, "reason": "hi"}')\
            ["score"] == 0
        assert R.parse_score_response("no json here at all") is None
        assert R.parse_score_response("") is None
        rambling = " ".join(["w"] * 30)
        out = R.parse_score_response(
            '{"score": 5, "reason": "' + rambling + '"}')
        assert len(out["reason"].split()) == 20  # <=20 words enforced

    def test_candidate_text_truncated_600(self, conn):
        c = R.load_candidate(conn, "decisions", 1, bm25_rank=0)
        assert len(c["text"]) <= 600
        assert c["key"] == "decisions:1"
        assert c["title_or_class"] == "major"
        cc = R.load_candidate(conn, "classics", 1, bm25_rank=0)
        assert len(cc["text"]) <= 600
        assert cc["title_or_class"] == "sunzi"


# ---------------------------------------------------------------- full API
class TestRerankAPI:
    def test_sorted_desc_and_topk_cutoff(self, conn, monkeypatch):
        monkeypatch.setattr(
            R, "score_candidates",
            lambda scenario, candidates, **kw: _ok_scores(candidates))
        res = R.rerank(SCENARIO_APERTURE, k=5, conn=conn, use_cache=False)
        assert len(res) == 5
        scores = [r["score"] for r in res]
        assert scores == sorted(scores, reverse=True)
        assert len({(r["source"], r["id"]) for r in res}) == 5

    def test_tiebreak_by_bm25_rank(self, conn, monkeypatch):
        monkeypatch.setattr(
            R, "score_candidates",
            lambda scenario, candidates, **kw:
                {c["key"]: {"score": 5, "reason": "tie", "ok": True}
                 for c in candidates})
        res = R.rerank(SCENARIO_APERTURE, k=8, conn=conn, use_cache=False)
        keys = [(r["bm25_rank"], r["source"]) for r in res]
        assert keys == sorted(keys), "ties must fall back to bm25 rank"

    def test_api_shape_and_snippet_len(self, conn, monkeypatch):
        monkeypatch.setattr(
            R, "score_candidates",
            lambda scenario, candidates, **kw: _ok_scores(candidates))
        res = R.rerank(SCENARIO_APERTURE, k=3, conn=conn, use_cache=False)
        assert res and len(res) == 3
        for r in res:
            assert set(r) == {"id", "source", "title_or_class", "score",
                              "reason", "text_snippet", "bm25_rank"}
            assert isinstance(r["score"], int) and 0 <= r["score"] <= 10
            assert 0 < len(r["text_snippet"]) <= 200
            assert len(r["reason"].split()) <= 20

    def test_sources_filter(self, conn, monkeypatch):
        monkeypatch.setattr(
            R, "score_candidates",
            lambda scenario, candidates, **kw: _ok_scores(candidates))
        res = R.rerank("war deception appear incapable when capable",
                       k=3, sources=("classics",), conn=conn,
                       use_cache=False)
        assert res and all(r["source"] == "classics" for r in res)

    def test_llm_failure_scores_zero_no_crash(self, conn, monkeypatch):
        def failing(scenario, candidates, **kw):
            return {c["key"]: {"score": 0, "reason": R.FAIL_REASON,
                               "ok": False} for c in candidates}
        monkeypatch.setattr(R, "score_candidates", failing)
        res = R.rerank(SCENARIO_APERTURE, k=3, conn=conn, use_cache=False)
        assert len(res) == 3
        assert all(r["score"] == 0 for r in res)
        assert all(r["reason"] == R.FAIL_REASON for r in res)

    def test_partial_llm_results_fill_zero(self, conn, monkeypatch):
        def partial(scenario, candidates, **kw):
            out = {}
            for i, c in enumerate(candidates):
                if i < 2:  # only first two candidates scored
                    out[c["key"]] = {"score": 9 - i, "reason": "hit",
                                     "ok": True}
            return out
        monkeypatch.setattr(R, "score_candidates", partial)
        res = R.rerank(SCENARIO_APERTURE, k=5, conn=conn, use_cache=False)
        assert len(res) == 5
        zeros = [r for r in res if r["score"] == 0]
        assert len(zeros) == 3
        assert all(r["reason"] == R.FAIL_REASON for r in zeros)

    def test_empty_scenario_returns_empty(self, conn):
        assert R.rerank("", k=3, conn=conn) == []
        assert R.rerank("   ", k=3, conn=conn) == []


# -------------------------------------------------------------------- cache
class TestCache:
    def test_cache_write_and_read(self, conn, monkeypatch, tmp_path):
        cache_dir = tmp_path / "rerank-cache"
        calls = {"n": 0}

        def counting_scorer(scenario, candidates, **kw):
            calls["n"] += 1
            return {c["key"]: {"score": 7, "reason": "mock cached",
                               "ok": True} for c in candidates}

        monkeypatch.setattr(R, "score_candidates", counting_scorer)
        r1 = R.rerank(SCENARIO_APERTURE, k=3, conn=conn,
                      cache_dir=cache_dir)
        assert calls["n"] == 1
        files = list(cache_dir.glob("*.json"))
        assert len(files) >= 3, "one cache file per scored candidate"
        # sha1(scenario + candidate_id) naming, 40 hex chars
        assert all(len(f.stem) == 40 for f in files)

        def must_not_call(scenario, candidates, **kw):
            raise AssertionError("LLM called despite warm cache")
        monkeypatch.setattr(R, "score_candidates", must_not_call)
        r2 = R.rerank(SCENARIO_APERTURE, k=3, conn=conn,
                      cache_dir=cache_dir)
        assert r1 == r2
        assert all(r["score"] == 7 and r["reason"] == "mock cached"
                   for r in r2)

    def test_cache_key_is_scenario_and_candidate_pair(self, tmp_path):
        p1 = R.cache_path("s", "decisions:1", tmp_path)
        p2 = R.cache_path("s2", "decisions:1", tmp_path)
        p3 = R.cache_path("s", "decisions:2", tmp_path)
        assert p1 != p2 and p1 != p3 and p2 != p3
        import hashlib
        assert p1.stem == hashlib.sha1(b"sdecisions:1").hexdigest()

    def test_no_cache_flag_skips_disk(self, conn, monkeypatch, tmp_path):
        cache_dir = tmp_path / "should-not-exist"
        monkeypatch.setattr(
            R, "score_candidates",
            lambda scenario, candidates, **kw:
                {c["key"]: {"score": 6, "reason": "x", "ok": True}
                 for c in candidates})
        R.rerank(SCENARIO_APERTURE, k=3, conn=conn,
                 cache_dir=cache_dir, use_cache=False)
        assert not cache_dir.exists()
