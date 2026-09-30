#!/usr/bin/env python3
"""Provenance-gated ingestion for user-provided novel files.

Design (from the project's legal posture):
- The file must be placed by the USER into data/inbox/ — the pipeline never
  fetches copyrighted material itself.
- The file is hashed, registered with provenance_status='user_responsibility'
  and stored in quarantine until explicitly confirmed with
  --confirm-user-provided (the user's own legal assertion).
- Once confirmed, chapters are segmented and ingested into corpus_items
  (item_type='chapter') for retrieval + extraction.

Usage:
  python ingest_local_corpus.py                 # list inbox files (quarantine)
  python ingest_local_corpus.py --confirm-user-provided FILE
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import db_utils  # noqa: E402

INBOX = Path(__file__).resolve().parents[2] / "data" / "inbox"

CHAPTER_PATTERNS = [
    re.compile(r"^第[零一二三四五六七八九十百千两0-9]+[章节][\s\S]{0,60}?$", re.M),
    re.compile(r"^Chapter\s+\d+.*$", re.M | re.I),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def segment_chapters(text: str) -> list[tuple[str, str]]:
    """Split into (chapter_title, body) using common CN/EN chapter markers."""
    marks: list[tuple[int, str]] = []
    for pat in CHAPTER_PATTERNS:
        for m in pat.finditer(text):
            marks.append((m.start(), m.group().strip()))
    marks = sorted(set(marks))
    if len(marks) < 2:
        return [("full_text", text.strip())]
    out = []
    for i, (pos, title) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[pos + len(title):end].strip()
        if len(body) > 100:
            out.append((title, body))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", help="file inside data/inbox/ to ingest")
    ap.add_argument("--confirm-user-provided", action="store_true",
                    help="assert you have lawful access; lifts quarantine")
    ap.add_argument("--title", default=None, help="work title")
    args = ap.parse_args()

    INBOX.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in INBOX.iterdir()
                   if p.is_file() and not p.name.startswith("."))
    if not args.file:
        print("inbox files (quarantined until confirmed):")
        for p in files:
            print(f"  {p.name}  sha256={sha256(p)[:16]}…")
        return

    target = INBOX / args.file
    if not target.exists():
        print(f"error: {target} not found (place files in data/inbox/)")
        sys.exit(1)

    digest = sha256(target)
    conn = db_utils.connect()
    db_utils.init_schema(conn)

    title = args.title or target.stem
    source_key = f"user_provided:{digest[:16]}"
    if not args.confirm_user_provided:
        sid = db_utils.ensure_source(
            conn, source_key=source_key, kind="user_provided_corpus",
            title=title, license_="User responsibility (unverified)",
            provenance_status="quarantined",
            meta={"sha256": digest, "size": target.stat().st_size})
        print(f"QUARANTINED. sha256={digest}")
        print("Review provenance, then re-run with --confirm-user-provided "
              "to assert lawful access and enable ingestion.")
        return

    sid = db_utils.ensure_source(
        conn, source_key=source_key, kind="user_provided_corpus",
        title=title, license_="User responsibility (asserted lawful access)",
        provenance_status="user_responsibility",
        meta={"sha256": digest, "size": target.stat().st_size,
              "confirmed_at": db_utils.utcnow()})
    text = target.read_text(encoding="utf-8", errors="replace")
    chapters = segment_chapters(text)
    for ch_title, body in chapters:
        db_utils.add_corpus_item(conn, sid, "chapter", ch_title, body,
                                 meta={"sha256": digest},
                                 tags=f"chapter {title}")
    print(f"[ok] ingested {len(chapters)} chapters from {target.name} "
          f"(sha256={digest[:16]}…, provenance=user_responsibility)")


if __name__ == "__main__":
    main()
