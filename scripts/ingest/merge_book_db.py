#!/usr/bin/env python3
"""Merge the user-provided Reverend Insanity book DB (their own upload of the
cleaned fan-translation corpus) into the master fang_yuan.db.

- source registered: kind=user_provided_corpus, provenance=user_responsibility
- chapters -> corpus_items(item_type='chapter') with FTS (rowid-aligned)
- chapter_notes -> corpus_items(item_type='chapter_note')
- volume map: approximate arc boundaries stored in sources.meta_json
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import db_utils  # noqa: E402

SRC_DB = Path(__file__).resolve().parents[2] / "data" / "inbox" / "reverend_insanity.db"

# Approximate volume/arc boundaries (fan translation convention; confidence
# medium — refine via extraction pass). Keyed by chapter ranges.
ARC_MAP = [
    (1, 230, "v1_qing_mao_mountain"),
    (231, 420, "v2_caravan_shang_city"),
    (421, 570, "v3_yi_tian_northern_plains"),
    (571, 760, "v4_northern_plains_imperial_court"),
    (761, 980, "v5_central_continent_zombie_ascension"),
    (981, 1180, "v6_ancient_soul_central_plains"),
    (1181, 1400, "v7_reversed_heaven_alliance"),
    (1401, 1600, "v8_western_desert_immortal_aperture"),
    (1601, 1820, "v9_reversed_flow_river_venerables"),
    (1821, 2063, "v10_crazed_demon_cave_endgame"),
]


def arc_for(ch: int) -> str:
    for lo, hi, name in ARC_MAP:
        if lo <= ch <= hi:
            return name
    return "unknown"


def main() -> None:
    src = sqlite3.connect(str(SRC_DB))
    src.row_factory = sqlite3.Row
    meta = {r["key"]: r["value"] for r in src.execute("SELECT key, value FROM metadata")}

    conn = db_utils.connect()
    db_utils.init_schema(conn)

    sid = db_utils.ensure_source(
        conn,
        source_key="user_provided:reverend_insanity_db",
        kind="user_provided_corpus",
        title=meta.get("book_title", "Reverend Insanity"),
        url="https://github.com/ssmurfgg04-gif/fang-yuan-strategic-system/commit/e9a5b3f",
        license_="User responsibility — user-uploaded fan-translation corpus "
                 "(Atlas Studios / Skyfarrow et al.); NOT redistributed by this repo",
        provenance_status="user_responsibility",
        meta={"sha256_gz": "e9a5b3f-upload", "chapter_count": meta.get("chapter_count"),
              "total_words": meta.get("total_words"),
              "total_chars": meta.get("total_chars"), "built_utc": meta.get("built_utc"),
              "schema_version": meta.get("schema_version"),
              "translator_note": "multiple translators; per-chapter translator column",
              "arc_map_confidence": "medium — approximate boundaries, refine later"})

    existing = {r[0] for r in conn.execute(
        "SELECT json_extract(meta_json,'$.chapter_id') FROM corpus_items "
        "WHERE source_id=? AND item_type='chapter'", (sid,))}

    n_ch, n_notes = 0, 0
    for row in src.execute(
            "SELECT chapter_id, title, body, word_count, char_count, translator, "
            "editor, sha256, epub_position FROM chapters ORDER BY chapter_id"):
        cid = row["chapter_id"]
        if cid in existing:
            continue
        tags = f"chapter novel {arc_for(cid)} {row['translator'] or ''}"
        db_utils.add_corpus_item(
            conn, sid, "chapter", f"ch{cid:04d} {row['title']}", row["body"],
            lang="en",
            meta={"chapter_id": cid, "word_count": row["word_count"],
                  "char_count": row["char_count"], "translator": row["translator"],
                  "editor": row["editor"], "sha256": row["sha256"],
                  "epub_position": row["epub_position"], "arc": arc_for(cid)},
            tags=tags)
        n_ch += 1

    for row in src.execute(
            "SELECT note_id, chapter_id, note_type, seq, note_text "
            "FROM chapter_notes ORDER BY note_id"):
        key = f"note{row['note_id']}"
        db_utils.add_corpus_item(
            conn, sid, "chapter_note", f"ch{row['chapter_id']:04d} note {row['seq']}",
            row["note_text"], lang="en",
            meta={"note_id": row["note_id"], "chapter_id": row["chapter_id"],
                  "note_type": row["note_type"]},
            tags=f"note {row['note_type']}")
        n_notes += 1

    src.close()
    print(f"[ok] merged {n_ch} chapters + {n_notes} notes "
          f"(provenance=user_responsibility)")
    total_words = sum(
        r[0] or 0 for r in conn.execute(
            "SELECT json_extract(meta_json,'$.word_count') FROM corpus_items "
            "WHERE item_type='chapter'"))
    print(f"[ok] total novel words in DB: {total_words:,}")


if __name__ == "__main__":
    main()
