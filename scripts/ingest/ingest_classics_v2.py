#!/usr/bin/env python3
"""Ingest classics expansion (catalog v2) — additional units for existing
works, deduped by unit_ref. Same provenance as catalog v1."""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import db_utils  # noqa: E402
from ingest.ingest_classics import api_get, fetch_page_text  # noqa: E402

import glob as _glob
_catalog_dir = Path(__file__).resolve().parents[1] / "seed" / "data"
CATALOGS = sorted(_glob.glob(str(_catalog_dir / "classics_catalog_v*.json")))


def main() -> None:
    conn = db_utils.connect()
    db_utils.init_schema(conn)

    sid = db_utils.ensure_source(
        conn, source_key="wikisource:zh_classics",
        kind="public_domain_classic",
        title="Chinese strategic classics (zh.wikisource)",
        url="https://zh.wikisource.org/",
        license_="Public domain (pre-1912 texts)",
        provenance_status="lawful_verified",
        meta={"expansion": "v2"})

    existing = {r[0] for r in conn.execute("SELECT unit_ref FROM classics")}
    fetched, missing = 0, []
    for cat_path in CATALOGS:
      catalog = json.loads(Path(cat_path).read_text(encoding="utf-8"))
      for group, spec in catalog.items():
        work = spec["work_key"]
        for unit in spec["units"]:
            ref = f"{spec['work_key']}·{unit['ref']}"
            work = spec["work_key"]
            if ref in existing:
                print(f"[skip] {ref}")
                continue
            text = fetch_page_text(unit["page"])
            if not text:
                missing.append((work, unit["page"]))
                continue
            db_utils.add_classic(
                conn, source_id=sid, work=work,
                unit_ref=ref, text_zh=text.strip(),
                text_en=unit.get("text_en"),
                translation_source=unit.get("translation_source"),
                analyst_summary=unit.get("analyst_summary"),
                principles=unit.get("principles"),
                fang_yuan_anchor=unit.get("fang_yuan_anchor"),
                passage_tag="CHINESE_STRATEGIC_TRADITION", lang="zh")
            db_utils.add_corpus_item(
                conn, sid, "classic_text", f"{work}:{unit['ref']}",
                text.strip(), lang="zh", meta={"work": work},
                tags=f"classic {work} {unit['ref']}")
            fetched += 1
            print(f"[ok] {ref} ({len(text)} chars)")
            time.sleep(1.2)

    print(f"\nfetched {fetched}; missing {len(missing)}")
    for w, p in missing:
        print(f"  MISSING: {w} :: {p}")


if __name__ == "__main__":
    main()
