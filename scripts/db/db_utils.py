"""Shared DB utilities for the Fang Yuan Strategic System.

Every write goes through here so FTS indexes stay in sync and provenance
is enforced: corpus rows MUST reference a registered source.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "db" / "fang_yuan.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()


# ------------------------------------------------------------------ sources
def ensure_source(conn, source_key, kind, title=None, url=None, license_=None,
                  provenance_status="lawful_verified", meta=None) -> int:
    """Register or update a source; return its id."""
    cur = conn.execute(
        "SELECT id FROM sources WHERE source_key=?", (source_key,))
    row = cur.fetchone()
    meta_json = json.dumps(meta, ensure_ascii=False) if meta else None
    if row:
        conn.execute(
            "UPDATE sources SET kind=?, title=?, url=?, license=?, "
            "provenance_status=?, meta_json=? WHERE id=?",
            (kind, title, url, license_, provenance_status, meta_json, row["id"]))
        conn.commit()
        return row["id"]
    cur = conn.execute(
        "INSERT INTO sources(source_key, kind, title, url, license, "
        "provenance_status, fetched_at, meta_json) VALUES (?,?,?,?,?,?,?,?)",
        (source_key, kind, title, url, license_, provenance_status,
         utcnow(), meta_json))
    conn.commit()
    return cur.lastrowid


# ------------------------------------------------------------- corpus + FTS
def _fts_insert(conn, fts_table: str, base_id: int, cols: tuple, vals: tuple):
    """Insert into contentless FTS with explicit rowid == base-table id so
    joins always align. NEVER use the 'delete' command on these tables; use
    scripts/db/reindex_fts.py to rebuild instead."""
    conn.execute(
        f"INSERT INTO {fts_table}(rowid, {','.join(cols)}) VALUES (?{',?' * len(cols)})",
        (base_id, *vals))


def add_corpus_item(conn, source_id, item_type, title, text, lang="en", meta=None,
                    tags="") -> int:
    cur = conn.execute(
        "INSERT INTO corpus_items(source_id, item_type, title, lang, text, meta_json) "
        "VALUES (?,?,?,?,?,?)",
        (source_id, item_type, title, lang, text,
         json.dumps(meta, ensure_ascii=False) if meta else None))
    _fts_insert(conn, "fts_corpus", cur.lastrowid,
                ("title", "text", "tags"), (title or "", text, tags))
    conn.commit()
    return cur.lastrowid


def add_finding(conn, topic, query, source_url, source_title, finding,
                confidence="medium") -> int:
    cur = conn.execute(
        "INSERT INTO findings(topic, query, source_url, source_title, finding, "
        "confidence, created_at) VALUES (?,?,?,?,?,?,?)",
        (topic, query, source_url, source_title, finding, confidence, utcnow()))
    _fts_insert(conn, "fts_findings", cur.lastrowid,
                ("topic", "finding"), (topic or "", finding or ""))
    conn.commit()
    return cur.lastrowid


def add_decision(conn, **kw) -> int:
    cols = ("event_id", "source_id", "record_class", "situation", "objective",
            "known_facts_json", "unknowns_json", "resources_json",
            "options_json", "selected_action", "rejected_options_json", "cost",
            "result_json", "lesson_json", "canon_anchor", "volume", "confidence")
    vals = [json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
            for v in (kw.get(c) for c in cols)]
    cur = conn.execute(
        f"INSERT INTO decisions({','.join(cols)}) "
        f"VALUES ({','.join('?' * len(cols))})", vals)
    lesson = kw.get("lesson")
    _fts_insert(conn, "fts_decisions", cur.lastrowid,
                ("situation", "selected_action", "lesson", "canon_anchor"),
                (kw.get("situation") or "", kw.get("selected_action") or "",
                 " ".join(lesson) if isinstance(lesson, list) else (lesson or ""),
                 kw.get("canon_anchor") or ""))
    conn.commit()
    return cur.lastrowid


def add_quote(conn, **kw) -> int:
    cols = ("text_en", "text_zh", "pinyin", "translation_kind", "speaker",
            "chapter_ref", "scene", "theme", "verification_status",
            "variants_json", "confidence", "source_id")
    vals = [json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
            for v in (kw.get(c) for c in cols)]
    cur = conn.execute(
        f"INSERT INTO quotes({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
        vals)
    _fts_insert(conn, "fts_quotes", cur.lastrowid,
                ("text_en", "text_zh", "theme", "speaker"),
                (kw.get("text_en") or "", kw.get("text_zh") or "",
                 kw.get("theme") or "", kw.get("speaker") or ""))
    conn.commit()
    return cur.lastrowid


def add_classic(conn, **kw) -> int:
    cols = ("source_id", "work", "unit_ref", "text_zh", "text_en",
            "translation_source", "analyst_summary", "principles_json",
            "fang_yuan_anchor", "passage_tag", "lang")
    vals = [json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
            for v in (kw.get(c) for c in cols)]
    cur = conn.execute(
        f"INSERT INTO classics({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
        vals)
    _fts_insert(conn, "fts_classics", cur.lastrowid,
                ("work", "text_zh", "text_en", "analyst_summary"),
                (kw.get("work") or "", kw.get("text_zh") or "",
                 kw.get("text_en") or "", kw.get("analyst_summary") or ""))
    conn.commit()
    return cur.lastrowid


def add_lesson(conn, phase, category, statement, detail=None, origin=None) -> int:
    cur = conn.execute(
        "INSERT INTO lessons(phase, category, statement, detail, origin, ts) "
        "VALUES (?,?,?,?,?,?)",
        (phase, category, statement, detail, origin, utcnow()))
    conn.commit()
    return cur.lastrowid


def log_test(conn, test_name, outcome, duration_ms=None, details=None) -> int:
    cur = conn.execute(
        "INSERT INTO test_runs(test_name, outcome, duration_ms, details_json, created_at) "
        "VALUES (?,?,?,?,?)",
        (test_name, outcome, duration_ms,
         json.dumps(details, ensure_ascii=False, default=str) if details else None,
         utcnow()))
    conn.commit()
    return cur.lastrowid


def log_simulator_run(conn, env, seed, horizon, agent, trajectory,
                      final_metrics) -> int:
    cur = conn.execute(
        "INSERT INTO simulator_runs(env, seed, horizon, agent, trajectory_json, "
        "final_metrics_json, created_at) VALUES (?,?,?,?,?,?,?)",
        (env, seed, horizon, agent,
         json.dumps(trajectory, ensure_ascii=False, default=str),
         json.dumps(final_metrics, ensure_ascii=False, default=str), utcnow()))
    conn.commit()
    return cur.lastrowid


# ------------------------------------------------------------------- query
_FTS_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with",
    "what", "why", "how", "who", "when", "did", "does", "do", "is", "are",
    "was", "were", "be", "been", "than", "that", "this", "it", "its", "as",
    "at", "by", "from", "not", "no", "yes", "rather", "simply", "should",
    "would", "could", "can", "must", "into", "about", "beyond", "over",
}


def sanitize_fts_query(q: str) -> str:
    """Make arbitrary user text safe for FTS5 MATCH: strip punctuation,
    drop stopwords, OR-join tokens for recall. CJK runs become quoted phrases."""
    import re as _re
    tokens = _re.findall(r"[\w\u4e00-\u9fff]+", q or "")
    out = []
    cjk_buf: list[str] = []
    for t in tokens:
        if _re.search(r"[\u4e00-\u9fff]", t):
            cjk_buf.append(t)
            continue
        if cjk_buf:
            out.append('"' + "".join(cjk_buf) + '"')
            cjk_buf = []
        if t.lower() not in _FTS_STOPWORDS and len(t) > 1:
            out.append(f'"{t.lower()}"')
    if cjk_buf:
        out.append('"' + "".join(cjk_buf) + '"')
    return " OR ".join(out) if out else '""'


def search(conn, table: str, q: str, limit: int = 10):
    """FTS search returning rowids only — contentless FTS5 tables do not
    store column values; join back to base tables by rowid."""
    assert table in {"fts_corpus", "fts_decisions", "fts_quotes",
                     "fts_classics", "fts_findings"}
    try:
        return conn.execute(
            f"SELECT rowid FROM {table} WHERE {table} MATCH ? "
            f"ORDER BY rank LIMIT ?",
            (sanitize_fts_query(q), limit)).fetchall()
    except sqlite3.OperationalError:
        return []


def stats(conn) -> dict:
    row = conn.execute("SELECT * FROM v_db_stats").fetchone()
    return dict(row) if row else {}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        conn = connect()
        init_schema(conn)
        print(f"schema initialized at {DB_PATH}")
        print(json.dumps(stats(conn), indent=2))
