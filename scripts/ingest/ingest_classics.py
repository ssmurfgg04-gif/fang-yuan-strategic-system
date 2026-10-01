#!/usr/bin/env python3
"""Ingest public-domain Chinese strategic classics from zh.wikisource.

All works are pre-1912 texts (public domain). Fetched via the permitted
MediaWiki API (action=query&prop=extracts, curl UA). Chinese originals go
into `classics` (text_zh) + `corpus_items`, tagged CHINESE_STRATEGIC_TRADITION.
Analyst summaries/principles/anchors are authored in classics_catalog.json.
Splitting: daodejing by 章 markers; sunzi by 篇 headings.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import db_utils  # noqa: E402

CATALOG = Path(__file__).resolve().parents[1] / "seed" / "data" / "classics_catalog.json"
CACHE = Path(__file__).resolve().parents[2] / "data" / "raw" / "wikisource_cache"
CN_NUM = "一二三四五六七八九十百零"


def cn_to_int(s: str) -> int | None:
    """Convert Chinese numeral like 三十六/一百二十九 to int (best effort)."""
    digits = {"零": 0, "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9}
    units = {"十": 10, "百": 100, "千": 1000}
    if not s or any(ch not in digits and ch not in units for ch in s):
        return None
    total, current = 0, 0
    for ch in s:
        if ch in digits:
            current = digits[ch]
        else:
            mult = units[ch]
            if current == 0:
                current = 1
            if mult == 10 and total % 1000 >= 100:  # e.g. 一百二十
                total += current * mult
            else:
                total += current * mult if total == 0 or mult >= 100 else current * mult
                if mult >= 100:
                    pass
            current = 0
    return total + current


def normalize_unit_key(work: str, ref: str) -> str:
    """'三十六章' -> '第36章' for authored-key matching (daodejing style)."""
    m = re.match(r"^(第?)([零一二三四五六七八九十百兩]+)章$", ref.strip())
    if m:
        n = cn_to_int(m.group(2))
        if n is not None:
            return f"第{n}章"
    return ref


def api_get(params: dict) -> dict:
    base = "https://zh.wikisource.org/w/api.php?"
    qs = urllib.parse.urlencode(params)
    for attempt in range(6):
        proc = subprocess.run(
            ["curl", "-s", "-m", "30", "-A", "curl/8.5.0", base + qs],
            capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip().startswith("{"):
            return json.loads(proc.stdout)
        time.sleep(min(20, 5 * (attempt + 1)))
    raise RuntimeError(f"wikisource API failed for {params}")


def fetch_page_text(title: str) -> str | None:
    """Fetch with on-disk cache so we never re-hit the API for the same page."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / (urllib.parse.quote(title, safe="") + ".txt")
    if cache_file.exists() and cache_file.stat().st_size > 50:
        txt = cache_file.read_text(encoding="utf-8")
        return None if txt == "__MISSING__" else txt
    j = api_get({"action": "query", "prop": "extracts", "explaintext": "1",
                 "titles": title, "format": "json", "redirects": "1"})
    result: str | None = None
    for page in j["query"]["pages"].values():
        if "missing" in page:
            result = None
        else:
            result = (page.get("extract") or "").strip() or None
    cache_file.write_text(result if result else "__MISSING__", encoding="utf-8")
    return result


def split_daodejing(text: str) -> list[tuple[str, str]]:
    """Split by '=== X章 ===' headings (匯校版 layout)."""
    pat = re.compile(rf"^===\s*(第?[{CN_NUM}]+章)\s*===\s*$", re.M)
    marks = [(m.start(), m.group(1)) for m in pat.finditer(text)]
    if not marks:
        return []
    units = []
    for i, (pos, mark) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        seg = text[pos:end].strip()
        if len(seg) > 20:
            units.append((mark, seg))
    return units


def split_sunzi(text: str) -> list[tuple[str, str]]:
    """Split by '== 篇第X ==' wiki headings."""
    pat = re.compile(r"^==\s*(\S+?)\s*==\s*$", re.M)
    marks = [(m.start(), m.group(1)) for m in pat.finditer(text)]
    if not marks:
        return [("full_text", text)]
    units = []
    if marks[0][0] > 120:
        units.append(("序", text[:marks[0][0]].strip()))
    for i, (pos, mark) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        seg = text[pos:end].strip()
        if len(seg) > 40:
            units.append((mark, seg))
    return units


def main() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    conn = db_utils.connect()
    db_utils.init_schema(conn)

    source_id = db_utils.ensure_source(
        conn, source_key="wikisource:zh_classics",
        kind="public_domain_classic",
        title="Chinese strategic classics (zh.wikisource)",
        url="https://zh.wikisource.org/",
        license_="Public domain (pre-1912 texts)",
        provenance_status="lawful_verified",
        meta={"works": list(catalog.keys())})

    fetched, missing = 0, []
    for work, spec in catalog.items():
        auth = spec.get("authored", {})
        for unit in spec["units"]:
            page = unit["page"]
            text = fetch_page_text(page)
            if not text:
                missing.append((work, page))
                continue

            # decide unit decomposition
            if spec.get("split") == "chapter":
                pieces = split_daodejing(text)
            elif spec.get("split") == "pian":
                pieces = split_sunzi(text)
            else:
                pieces = [(unit.get("ref") or page, text)]

            # authored summary lookup for split units: match by key substring
            for ref, seg in pieces:
                summary = principles = anchor = None
                keys_to_try = {ref, normalize_unit_key(work, ref)}
                matched = None
                if ref in auth:
                    matched = auth[ref]
                else:
                    for key in keys_to_try:
                        if key in auth:
                            matched = auth[key]
                            break
                    if matched is None:
                        for key, a in auth.items():
                            if key and key in ref:
                                matched = a
                                break
                if matched:
                    summary = matched.get("analyst_summary")
                    principles = matched.get("principles")
                    anchor = matched.get("fang_yuan_anchor")

                db_utils.add_classic(
                    conn, source_id=source_id, work=work,
                    unit_ref=f"{work}·{ref}", text_zh=seg,
                    text_en=unit.get("text_en"),
                    translation_source=unit.get("translation_source"),
                    analyst_summary=summary, principles=principles,
                    fang_yuan_anchor=anchor,
                    passage_tag="CHINESE_STRATEGIC_TRADITION", lang="zh")
                db_utils.add_corpus_item(
                    conn, source_id, "classic_text", f"{work}:{ref}",
                    seg, lang="zh", meta={"work": work},
                    tags=f"classic {work} {ref}")
                fetched += 1
            print(f"[ok] {work} :: {page} -> {len(pieces)} units")
            time.sleep(1.0)

    print(f"\nfetched {fetched} units; missing {len(missing)}")
    for w, p in missing:
        print(f"  MISSING: {w} :: {p}")
    print(json.dumps(db_utils.stats(conn), indent=2))


if __name__ == "__main__":
    main()
