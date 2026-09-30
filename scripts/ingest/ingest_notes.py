#!/usr/bin/env python3
"""Ingest uploaded project notes -> corpus + lessons + plan_steps.
Also ingests research search results -> findings."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import db_utils  # noqa: E402

NOTES = Path("/home/z/my-project/upload/Notes_260930_224747.txt")
RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
SEED = Path(__file__).resolve().parents[1] / "seed" / "data" / "lessons_seed.json"


def main() -> None:
    conn = db_utils.connect()
    db_utils.init_schema(conn)

    # --- uploaded notes -> corpus chunks ---
    if NOTES.exists():
        sid = db_utils.ensure_source(
            conn, source_key="uploaded:project_notes",
            kind="uploaded_notes", title="Project planning notes (user conversation history)",
            license_="User's own notes", provenance_status="lawful_verified")
        text = NOTES.read_text(encoding="utf-8", errors="replace")
        chunks = [c.strip() for c in text.split("───") if len(c.strip()) > 120]
        for i, chunk in enumerate(chunks):
            db_utils.add_corpus_item(conn, sid, "note_chunk",
                                     f"notes_chunk_{i:02d}", chunk,
                                     meta={"chars": len(chunk)},
                                     tags="notes plan policy training benchmark")
        print(f"[ok] notes: {len(chunks)} chunks")

    # --- lessons + plan ---
    seed = json.loads(SEED.read_text(encoding="utf-8"))
    n_lessons = 0
    for L in seed["lessons"]:
        exists = conn.execute("SELECT 1 FROM lessons WHERE statement=?",
                              (L["statement"],)).fetchone()
        if exists:
            continue
        db_utils.add_lesson(conn, L.get("phase"), L.get("category"),
                            L["statement"], L.get("detail"), L.get("origin"))
        n_lessons += 1
    print(f"[ok] lessons inserted: {n_lessons}")

    for P in seed["plan"]:
        exists = conn.execute("SELECT 1 FROM plan_steps WHERE phase=? AND objective=?",
                              (P["phase"], P["objective"])).fetchone()
        if exists:
            continue
        conn.execute(
            "INSERT INTO plan_steps(phase, objective, deliverables_json, status, notes) "
            "VALUES (?,?,?,?,?)",
            (P["phase"], P["objective"], json.dumps(P["deliverables"], ensure_ascii=False),
             P.get("status", "pending"), P.get("notes")))
    conn.commit()
    print(f"[ok] plan steps: {conn.execute('SELECT COUNT(*) FROM plan_steps').fetchone()[0]}")

    # --- search results -> findings ---
    n_findings = 0
    for f in sorted(RAW.glob("search_*.json")):
        try:
            items = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        topic = f.stem.replace("search_", "")
        for it in items:
            finding = (it.get("snippet") or "").strip()
            if not finding:
                continue
            exists = conn.execute(
                "SELECT 1 FROM findings WHERE topic=? AND source_url=?",
                (topic, it.get("url", ""))).fetchone()
            if exists:
                continue
            db_utils.add_finding(conn, topic, "", it.get("url", ""),
                                 it.get("name", ""), finding)
            n_findings += 1
    print(f"[ok] findings inserted: {n_findings}")
    print(json.dumps(db_utils.stats(conn), indent=2))


if __name__ == "__main__":
    main()
