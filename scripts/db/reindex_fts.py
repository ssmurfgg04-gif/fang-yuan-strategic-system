#!/usr/bin/env python3
"""Rebuild all contentless FTS5 indexes from base tables with explicit
rowid = base-table id, guaranteeing alignment forever."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import db_utils  # noqa: E402

# (fts_table, base_table, [(fts_col, base_col), ...])  base_col may be None
# for derived channels.
JOBS = [
    ("fts_corpus", "corpus_items",
     [("title", "title"), ("text", "text"), ("tags", "@tags@")]),
    ("fts_decisions", "decisions",
     [("situation", "situation"), ("selected_action", "selected_action"),
      ("lesson", "lesson_json"), ("canon_anchor", "canon_anchor")]),
    ("fts_quotes", "quotes",
     [("text_en", "text_en"), ("text_zh", "text_zh"),
      ("theme", "theme"), ("speaker", "speaker")]),
    ("fts_classics", "classics",
     [("work", "work"), ("text_zh", "text_zh"), ("text_en", "text_en"),
      ("analyst_summary", "analyst_summary")]),
    ("fts_findings", "findings",
     [("topic", "topic"), ("finding", "finding")]),
]


def derive(base: str, fts_col: str, base_col: str, row: sqlite3.Row) -> str:
    if base_col == "@tags@":
        tag = row["item_type"] or ""
        meta = row["meta_json"] if "meta_json" in row.keys() else None
        if meta:
            try:
                m = json.loads(meta)
                if isinstance(m, dict) and m.get("tags"):
                    tag += " " + str(m["tags"])
            except Exception:  # noqa: BLE001
                pass
        return tag
    v = row[base_col] if base_col in row.keys() else None
    if v is None:
        return ""
    if fts_col == "lesson":  # lesson_json may be a JSON list
        try:
            parsed = json.loads(v)
            v = " ".join(parsed) if isinstance(parsed, list) else str(parsed)
        except Exception:  # noqa: BLE001
            pass
    return str(v)


def main() -> None:
    conn = db_utils.connect()
    for fts, base, pairs in JOBS:
        # wipe contentless table
        for rid in [r[0] for r in conn.execute(f"SELECT rowid FROM {fts}")]:
            blanks = ",".join("''" for _ in pairs)
            conn.execute(
                f"INSERT INTO {fts}({fts}, rowid, "
                f"{','.join(c for c, _ in pairs)}) VALUES('delete', ?, {blanks})",
                (rid,))
        # reinsert aligned to base ids
        base_cols = sorted({bc for _, bc in pairs} - {"@tags@"})
        sel = f"id, {','.join(base_cols)}"
        if base == "corpus_items":
            sel += ", meta_json, item_type"
        n = 0
        for row in conn.execute(f"SELECT {sel} FROM {base}"):
            fts_cols = ",".join(c for c, _ in pairs)
            vals = [row["id"]] + [derive(base, fc, bc, row) for fc, bc in pairs]
            conn.execute(
                f"INSERT INTO {fts}(rowid, {fts_cols}) VALUES (?{',?' * len(pairs)})",
                vals)
            n += 1
        conn.commit()
        print(f"[ok] {fts} rebuilt: {n} rows")
    print("done")


if __name__ == "__main__":
    main()
