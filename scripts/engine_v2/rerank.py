#!/usr/bin/env python
"""Two-stage retrieval + LLM cross-reranker (engine v2).

Pipeline
--------
Stage 1 — cheap recall: FTS5 bm25 search (via scripts/db/db_utils.search,
query sanitized for the contentless indexes, join back by rowid) over the
``decisions`` and ``classics`` tables. If FTS returns fewer than 5
candidates overall, a deterministic keyword LIKE-scan fallback runs.
~20 candidates are interleaved by within-source bm25 rank so neither
source starves the other (bm25 scores are NOT comparable across tables,
only rank order is).

Stage 2 — LLM cross-rerank: every candidate (text truncated to ~600
chars) is scored 0-10 for *strategic-intent relevance* against the
scenario by a GLM model. Calls are batched through
``bun run llm_batch.mjs`` (z-ai-web-dev-sdk, concurrency 5, 60 s per-call
timeout, thinking disabled); if bun is unavailable the module falls back
to the ``z-ai chat`` CLI with a 4-worker thread pool. Up to 3 attempts
with backoff; failures are skipped and scored 0 — never fatal.

Responses must be strict JSON ``{"score": int 0-10, "reason": "<=20 words"}``;
anything else is parsed leniently and clamped.

A disk cache ``data/cache/rerank/<sha1(scenario + candidate_id)>.json``
makes identical (scenario, candidate) pairs free — same query, zero LLM
calls. Ordering is deterministic: score desc, then within-source bm25
rank, then source, then id.

Library API
-----------
    from engine_v2 import rerank
    top3 = rerank("scenario text", k=3)   # [{id, source, title_or_class,
                                          #   score, reason, text_snippet, ...}]

CLI
---
    python scripts/engine_v2/rerank.py --scenario "..." --k 3 \
        [--sources decisions,classics] [--out out.json] [--no-cache]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

try:  # package import (scripts/ on sys.path) …
    from db import db_utils
except ImportError:  # … or direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    # purge a half-initialized namespace 'db' (repo-root db/ DATA directory
    # can shadow the scripts/db package when cwd == repo root)
    for _m in [m for m in sys.modules if m == "db" or m.startswith("db.")]:
        del sys.modules[_m]
    from db import db_utils  # type: ignore

# --------------------------------------------------------------- constants
SOURCES = ("decisions", "classics")
DEFAULT_RECALL_SIZE = 20
MIN_FTS_FOR_NO_FALLBACK = 5
CANDIDATE_TEXT_CHARS = 600          # candidate text cap inside the LLM prompt
SCENARIO_CHARS = 900                # scenario cap inside the LLM prompt
SNIPPET_CHARS = 200                 # text_snippet length in results
DEFAULT_K = 3
LLM_CONCURRENCY = 5                 # bun batch workers (spec: 4-6)
CLI_CONCURRENCY = 4                 # z-ai CLI fallback workers
LLM_TIMEOUT_S = 60                  # per-call timeout
MAX_ATTEMPTS = 3                    # 1 call + 2 retries
BACKOFF_S = (2.0, 5.0)              # sleep before retry attempt 2 / 3
RATE_LIMIT_BACKOFF_S = (20.0, 40.0)  # longer cooldown when API says 429
SCORE_FLOOR, SCORE_CEIL = 0, 10
REASON_MAX_WORDS = 20
FAIL_REASON = "scoring unavailable"

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
BATCH_MJS = _HERE / "llm_batch.mjs"

SYSTEM_PROMPT = (
    "You are the relevance scorer inside the Fang Yuan strategic decision "
    "engine. You judge how well a candidate knowledge unit (a past decision "
    "record, or a classic Chinese strategy text unit) matches the strategic "
    "INTENT of a new scenario. Score 0-10: 10 = same strategic dilemma and "
    "intent; 7-9 = strongly transferable strategic pattern; 4-6 = partially "
    "related; 1-3 = weak/topical overlap only; 0 = unrelated. Judge the "
    "underlying strategy (concealment, exclusivity traps, exposure risk, "
    "third-party leverage, resource compounding, timing), not surface "
    "vocabulary. Respond with STRICT JSON only: "
    '{"score": <int 0-10>, "reason": "<=20 words"}. '
    "No markdown fences, no extra keys, no commentary."
)

_LIKE_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "your", "you", "from",
    "into", "about", "over", "under", "while", "when", "what", "why",
    "would", "could", "will", "must", "their", "them", "they", "than",
    "then", "have", "has", "had", "are", "was", "were", "but", "not",
    "all", "any", "may", "its", "his", "her", "one", "two", "other",
    "which", "there", "here", "been", "also", "more", "most", "some",
    "such", "only", "very", "upon", "after", "before", "between",
}

# ------------------------------------------------------------------ helpers


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_cache_dir() -> Path:
    env = os.environ.get("FANGYUAN_RERANK_CACHE_DIR")
    if env:
        return Path(env)
    return _REPO_ROOT / "data" / "cache" / "rerank"


def _clip(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "\u2026"


def cache_path(scenario: str, candidate_id: str,
               cache_dir: Path | None = None) -> Path:
    """Disk-cache path for one (scenario, candidate) pair."""
    digest = hashlib.sha1(
        (scenario + candidate_id).encode("utf-8")).hexdigest()
    base = Path(cache_dir) if cache_dir else default_cache_dir()
    return base / f"{digest}.json"


def _read_cache(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or "score" not in data:
        return None
    return data


def _write_cache(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass  # cache is best-effort; never break a run over it


# -------------------------------------------------- stage 1: cheap recall

def _candidate_text(conn, source: str, rid: int) -> tuple[str, str]:
    """Return (title_or_class, full_text) for one candidate row."""
    if source == "decisions":
        row = conn.execute(
            "SELECT record_class, situation, objective, selected_action, "
            "cost, canon_anchor FROM decisions WHERE id=?", (rid,)).fetchone()
        if row is None:
            return "", ""
        parts = [row["situation"] or ""]
        if row["objective"]:
            parts.append("objective: " + row["objective"])
        if row["selected_action"]:
            parts.append("action taken: " + row["selected_action"])
        if row["cost"]:
            parts.append("cost: " + row["cost"])
        if row["canon_anchor"]:
            parts.append("canon anchor: " + row["canon_anchor"])
        return row["record_class"] or "decision", " | ".join(parts)
    # classics
    row = conn.execute(
        "SELECT work, unit_ref, text_zh, analyst_summary, fang_yuan_anchor "
        "FROM classics WHERE id=?", (rid,)).fetchone()
    if row is None:
        return "", ""
    head = " ".join(x for x in (row["work"], row["unit_ref"]) if x)
    parts = [head]
    if row["analyst_summary"]:
        parts.append(row["analyst_summary"])
    if row["text_zh"]:
        parts.append(row["text_zh"])
    if row["fang_yuan_anchor"]:
        parts.append("maps to: " + row["fang_yuan_anchor"])
    return row["work"] or "classic", " | ".join(parts)


def load_candidate(conn, source: str, rid: int, bm25_rank: int) -> dict:
    title_or_class, text = _candidate_text(conn, source, rid)
    return {
        "id": rid,
        "source": source,
        "key": f"{source}:{rid}",
        "title_or_class": title_or_class,
        "text": _clip(text, CANDIDATE_TEXT_CHARS),
        "bm25_rank": bm25_rank,
    }


def recall_candidates(conn, scenario: str,
                      sources: tuple[str, ...] = SOURCES,
                      recall_size: int = DEFAULT_RECALL_SIZE) -> list[dict]:
    """Stage 1: top `recall_size` candidates by bm25, interleaved across
    sources by within-source rank; LIKE-scan fallback when FTS is thin."""
    sources = tuple(s for s in sources if s in SOURCES) or SOURCES
    per_source: dict[str, list[int]] = {}
    fts_total = 0
    for src in sources:
        fts_table = "fts_decisions" if src == "decisions" else "fts_classics"
        rows = db_utils.search(conn, fts_table, scenario, limit=recall_size)
        per_source[src] = [r[0] for r in rows]
        fts_total += len(per_source[src])

    candidates: list[dict] = []
    seen: set[tuple[str, int]] = set()

    def _add(src: str, rid: int, rank: int) -> None:
        if (src, rid) in seen or len(candidates) >= recall_size:
            return
        seen.add((src, rid))
        candidates.append(load_candidate(conn, src, rid, rank))

    if fts_total:
        max_len = max(len(v) for v in per_source.values())
        for rank in range(max_len):          # interleave by bm25 position
            for src in sources:
                lst = per_source[src]
                if rank < len(lst):
                    _add(src, lst[rank], rank)
            if len(candidates) >= recall_size:
                break

    if len(candidates) < MIN_FTS_FOR_NO_FALLBACK:
        fallback = _like_fallback(conn, scenario, sources, recall_size)
        for i, (src, rid) in enumerate(fallback):
            _add(src, rid, 1000 + i)        # sorts after any FTS hit

    return candidates


def _like_fallback(conn, scenario: str, sources: tuple[str, ...],
                   limit: int) -> list[tuple[str, int]]:
    """Deterministic keyword scan used when FTS recall is < 5."""
    tokens = [t.lower() for t in re.findall(r"[A-Za-z]{4,}", scenario or "")]
    tokens = [t for t in tokens if t not in _LIKE_STOPWORDS]
    tokens = sorted(dict.fromkeys(tokens), key=lambda t: (-len(t), t))[:6]
    if not tokens:
        return []
    scored: list[tuple[int, str, int]] = []
    for src in sources:
        if src == "decisions":
            rows = conn.execute(
                "SELECT id, situation, objective, selected_action, cost, "
                "canon_anchor FROM decisions").fetchall()
            blobf = lambda r: " ".join(  # noqa: E731
                r[c] or "" for c in
                ("situation", "objective", "selected_action", "cost",
                 "canon_anchor"))
        else:
            rows = conn.execute(
                "SELECT id, work, unit_ref, text_zh, analyst_summary, "
                "fang_yuan_anchor FROM classics").fetchall()
            blobf = lambda r: " ".join(  # noqa: E731
                r[c] or "" for c in
                ("work", "unit_ref", "text_zh", "analyst_summary",
                 "fang_yuan_anchor"))
        for r in rows:
            blob = blobf(r).lower()
            hits = sum(1 for t in tokens if t in blob)
            if hits:
                scored.append((hits, src, r["id"]))
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    return [(src, rid) for _, src, rid in scored[:limit]]


# --------------------------------------------- stage 2: LLM cross-rerank

def build_user_prompt(scenario: str, cand: dict) -> str:
    return (
        f"SCENARIO:\n{_clip(scenario, SCENARIO_CHARS)}\n\n"
        f"CANDIDATE [{cand['key']}] ({cand['title_or_class']}):\n"
        f"{cand['text']}\n\n"
        'Task: score the strategic-intent relevance of this candidate to '
        'the scenario. Respond with strict JSON '
        '{"score": <int 0-10>, "reason": "<=20 words"}.'
    )


def parse_score_response(content: str | None) -> dict | None:
    """Leniently extract {"score": int, "reason": str}; None on garbage."""
    if not content:
        return None
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict) or "score" not in data:
        return None
    try:
        score = int(round(float(data["score"])))
    except (TypeError, ValueError):
        return None
    score = max(SCORE_FLOOR, min(SCORE_CEIL, score))
    reason = data.get("reason")
    if not isinstance(reason, str):
        reason = ""
    return {"score": score, "reason": _clamp_reason_words(reason)}


def _clamp_reason_words(reason: str) -> str:
    """Enforce the '<=20 words' contract even if the model rambles."""
    return " ".join(reason.split()[:REASON_MAX_WORDS])


def _call_bun_batch(items: list[dict], timeout_s: int,
                    concurrency: int) -> dict[str, dict]:
    """One bun batch round. Returns {key: {ok, content|error}}; raises on
    infrastructure failure (missing bun, bad exit, unreadable output).

    The .mjs is executed from a neutral temp dir: if the repo sits under a
    directory that contains a node_modules without the SDK (observed at
    /home/z/node_modules), bun's module resolution stops there and never
    reaches its global cache; a /tmp ancestry has no node_modules, so bun
    auto-resolves z-ai-web-dev-sdk from its global install."""
    if shutil.which("bun") is None:
        raise RuntimeError("bun not on PATH")
    if not BATCH_MJS.exists():
        raise RuntimeError(f"missing {BATCH_MJS}")
    payload = {"items": items, "concurrency": concurrency,
               "timeout_ms": timeout_s * 1000}
    with tempfile.TemporaryDirectory(prefix="fy_rerank_") as td:
        tdp = Path(td)
        mjs_copy = tdp / "llm_batch.mjs"
        mjs_copy.write_text(BATCH_MJS.read_text(encoding="utf-8"),
                            encoding="utf-8")
        in_path = tdp / "in.json"
        out_path = tdp / "out.json"
        in_path.write_text(json.dumps(payload, ensure_ascii=False),
                           encoding="utf-8")
        batch_wall = timeout_s * max(1, -(-len(items) // concurrency)) + 60
        proc = subprocess.run(
            ["bun", "run", str(mjs_copy), str(in_path), str(out_path)],
            capture_output=True, text=True, timeout=batch_wall,
            cwd=str(tdp))
        if proc.returncode != 0 or not out_path.exists():
            raise RuntimeError(
                f"llm_batch.mjs failed rc={proc.returncode} "
                f"err={proc.stderr[-300:]}")
        data = json.loads(out_path.read_text(encoding="utf-8"))
    results = data.get("results", {})
    if not isinstance(results, dict):
        raise RuntimeError("llm_batch.mjs returned malformed results")
    return results


def _score_one_via_cli(system: str, user: str, timeout_s: int) -> str | None:
    """Single call through the `z-ai chat` CLI; returns content or None."""
    if shutil.which("z-ai") is None:
        return None
    fd, out_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        proc = subprocess.run(
            ["z-ai", "chat", "-p", user, "-s", system, "-o", out_path],
            capture_output=True, text=True, timeout=timeout_s + 15)
        if proc.returncode != 0:
            return None
        data = json.loads(Path(out_path).read_text(encoding="utf-8"))
        return data["choices"][0]["message"]["content"]
    except (subprocess.TimeoutExpired, OSError, ValueError,
            KeyError, IndexError):
        return None
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def _score_via_cli(items: list[dict], timeout_s: int) -> dict[str, dict]:
    """CLI fallback: thread pool of 4, each item retried by the caller's
    rounds — here one round = one call attempt per item."""
    def _one(item: dict) -> tuple[str, dict]:
        content = _score_one_via_cli(item["system"], item["user"], timeout_s)
        if content is None:
            return item["key"], {"ok": False, "error": "cli call failed"}
        return item["key"], {"ok": True, "content": content}

    out: dict[str, dict] = {}
    if not items:
        return out
    with ThreadPoolExecutor(max_workers=CLI_CONCURRENCY) as ex:
        for key, res in ex.map(_one, items):
            out[key] = res
    return out


def score_candidates(scenario: str, candidates: list[dict],
                     timeout_s: int = LLM_TIMEOUT_S,
                     concurrency: int = LLM_CONCURRENCY,
                     max_attempts: int = MAX_ATTEMPTS) -> dict[str, dict]:
    """Stage 2 for a batch of candidate dicts.

    Returns {key: {"score": int, "reason": str, "ok": bool}}. Every
    candidate gets an entry; failures score 0 with FAIL_REASON.
    This is the LLM boundary — tests monkeypatch it.
    """
    items = [{"key": c["key"],
              "system": SYSTEM_PROMPT,
              "user": build_user_prompt(scenario, c)}
             for c in candidates]
    ok_results: dict[str, dict] = {}
    pending = items
    rate_limited = False
    for attempt in range(max_attempts):
        if not pending:
            break
        if attempt:
            backoff = (RATE_LIMIT_BACKOFF_S if rate_limited else BACKOFF_S)
            time.sleep(backoff[min(attempt - 1, len(backoff) - 1)])
        try:
            round_results = _call_bun_batch(pending, timeout_s, concurrency)
        except Exception:  # infra failure → CLI fallback for this round
            round_results = _score_via_cli(pending, timeout_s)
        still_pending: list[dict] = []
        errors: list[str] = []
        for item in pending:
            res = round_results.get(item["key"])
            if res and res.get("ok"):
                parsed = parse_score_response(res.get("content"))
                if parsed is not None:
                    ok_results[item["key"]] = {**parsed, "ok": True}
                    continue
            errors.append(str((res or {}).get("error", "")))
            still_pending.append(item)
        rate_limited = any("429" in e or "too many requests" in e.lower()
                           for e in errors)
        pending = still_pending
    for item in pending:  # exhausted attempts → skip, score 0
        ok_results[item["key"]] = {"score": 0, "reason": FAIL_REASON,
                                   "ok": False}
    return ok_results


# ------------------------------------------------------------ public API

def rerank(scenario: str, k: int = DEFAULT_K,
           sources: tuple[str, ...] = SOURCES,
           recall_size: int = DEFAULT_RECALL_SIZE,
           conn=None, cache_dir: Path | None = None,
           use_cache: bool = True) -> list[dict]:
    """Full pipeline: recall → (cache-aware) LLM cross-rerank → strict top-k.

    Returns [{id, source, title_or_class, score, reason, text_snippet,
    bm25_rank}] sorted by score desc, then bm25 rank (deterministic
    tie-break), then source/id.
    """
    if not scenario or not scenario.strip():
        return []
    k = max(0, int(k))
    own_conn = conn is None
    conn = conn or db_utils.connect()
    try:
        candidates = recall_candidates(conn, scenario, sources, recall_size)
        if not candidates:
            return []

        scored: dict[str, dict] = {}
        missing: list[dict] = []
        for cand in candidates:
            entry = None
            if use_cache:
                entry = _read_cache(
                    cache_path(scenario, cand["key"], cache_dir))
            if entry is not None:
                scored[cand["key"]] = {"score": int(entry["score"]),
                                       "reason": str(entry.get("reason", "")),
                                       "ok": True}
            else:
                missing.append(cand)

        if missing:
            results = score_candidates(scenario, missing)
            for cand in missing:
                res = results.get(
                    cand["key"], {"score": 0, "reason": FAIL_REASON,
                                  "ok": False})
                scored[cand["key"]] = {"score": int(res.get("score", 0)),
                                       "reason": str(res.get("reason", "")),
                                       "ok": bool(res.get("ok", True))}
                if (use_cache and res.get("ok", True)
                        and int(res.get("score", 0)) > 0):
                    _write_cache(
                        cache_path(scenario, cand["key"], cache_dir),
                        {"key": cand["key"], "score": int(res["score"]),
                         "reason": res.get("reason", ""),
                         "model": "glm(z-ai)", "cached_at": _utcnow()})
    finally:
        if own_conn:
            conn.close()

    ranked = []
    for cand in candidates:
        s = scored.get(cand["key"], {"score": 0, "reason": FAIL_REASON})
        ranked.append({
            "id": cand["id"],
            "source": cand["source"],
            "title_or_class": cand["title_or_class"],
            "score": int(s["score"]),
            "reason": _clamp_reason_words(s["reason"]),
            "text_snippet": _clip(cand["text"], SNIPPET_CHARS),
            "bm25_rank": cand["bm25_rank"],
        })
    ranked.sort(key=lambda r: (-r["score"], r["bm25_rank"], r["source"],
                               r["id"]))
    return ranked[:k]


# ------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Two-stage retrieval + LLM cross-reranker "
                    "(strict top-k injection).")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--sources", default="decisions,classics",
                    help="comma list: decisions,classics")
    ap.add_argument("--out", default=None, help="also write JSON here")
    ap.add_argument("--recall-size", type=int, default=DEFAULT_RECALL_SIZE)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)

    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    results = rerank(
        args.scenario, k=args.k, sources=sources,
        recall_size=args.recall_size,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        use_cache=not args.no_cache)
    payload = {"scenario": args.scenario, "k": args.k,
               "sources": list(sources), "results": results}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
