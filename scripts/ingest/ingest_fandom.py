#!/usr/bin/env python3
"""Ingest fetched Reverend Insanity Fandom wiki pages into the master DB.

Source: reverend-insanity.fandom.com — content licensed CC BY-SA 3.0.
Ingestion uses the official MediaWiki API format (api.php), attribution kept
in sources table and README. Registers provenance, stores full article text,
extracts structured rows (characters, factions, gu_worms) from infoboxes.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import db_utils  # noqa: E402

from bs4 import BeautifulSoup  # noqa: E402

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "fandom"

# which pages map to which structured table
CHARACTER_PAGES = {"fang_yuan": "Fang Yuan", "bai_ning_bing": "Bai Ning Bing",
                   "fang_zheng": "Gu Yue Fang Zheng", "shang_yan_fei": "Shang Yan Fei"}
FACTION_PAGES = {"gu_yue_clan": "Gu Yue Clan", "heavenly_court": "Heavenly Court"}
GU_PAGES = {"liquor_worm": "Liquor Worm", "spring_autumn_cicada": "Spring Autumn Cicada"}


_NOISE = re.compile(
    r"^(sign in|create a free account|advertisement|skip to content|"
    r"don't have an account\?|reverend insanity wiki|fandom$|"
    r"pages? with broken file links|in:|explore|wiki activity|"
    r"community page|all pages|recent blog posts|random page|"
    r"fan feed|register|watchlist|history|talk \(\d+\)|edit|"
    r"visualeditor|categories|languages)$", re.I)


def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "figure", "img", "nav", "footer",
                     "aside", "table"]):
        tag.decompose()
    text = soup.get_text("\n")
    lines = [ln.strip() for ln in text.splitlines()]
    out, blank = [], 0
    for ln in lines:
        if not ln:
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        if _NOISE.match(ln):
            continue
        out.append(ln)
    return "\n".join(out).strip()


def parse_infobox(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    aside = soup.find("aside", class_="portable-infobox")
    if not aside:
        return {}
    data = {}
    title_el = aside.find("h2", class_=re.compile("pi-title"))
    if title_el:
        data["title"] = title_el.get_text(" ", strip=True)
    for item in aside.find_all("div", class_="pi-item", attrs={"data-source": True}):
        key = item["data-source"]
        label = item.find("h3", class_="pi-data-label")
        value = item.find("div", class_="pi-data-value")
        if label and value:
            data[key] = value.get_text(" ", strip=True)
    return data


def parse_sections(html: str) -> list[tuple[str, str]]:
    """Return [(section_title, text)] using h2/h3 boundaries."""
    soup = BeautifulSoup(html, "lxml")
    content = soup.find("div", class_="mw-parser-output") or soup
    sections = []
    cur_title = "introduction"
    buf: list[str] = []
    for el in content.find_all(["h2", "h3", "p", "ul", "ol"], recursive=True):
        if el.name in ("h2", "h3"):
            if buf:
                sections.append((cur_title, "\n".join(buf)))
            cur_title = el.get_text(" ", strip=True) or "section"
            buf = []
        else:
            t = el.get_text(" ", strip=True)
            if t:
                buf.append(t)
    if buf:
        sections.append((cur_title, "\n".join(buf)))
    # drop tiny/duplicate nav sections
    return [(t, x) for t, x in sections if len(x) > 80]


def main() -> None:
    conn = db_utils.connect()
    db_utils.init_schema(conn)
    n_files = 0
    for f in sorted(RAW.glob("*.json")):
        key = f.stem
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
            data = payload.get("data", payload)
            html = data.get("html", "") or ""
            if len(html) < 2000:
                print(f"  !! thin page skipped: {key}")
                continue
        except Exception as exc:  # noqa: BLE001
            print(f"  !! parse error {key}: {exc}")
            continue

        title = data.get("title") or key
        url = data.get("url") or f"https://reverend-insanity.fandom.com/wiki/{key}"
        text = clean_text(html)
        sections = parse_sections(html)
        ib = parse_infobox(html)

        source_id = db_utils.ensure_source(
            conn,
            source_key=f"fandom:{key}",
            kind="fandom_wiki",
            title=title,
            url=url,
            license_="CC BY-SA 3.0 (Fandom wiki article, attributed; "
                     "text ingested via permitted MediaWiki API format)",
            provenance_status="lawful_verified",
            meta={"sections": len(sections), "infobox_keys": list(ib)[:20]},
        )

        # full article → corpus
        tags = " ".join(t.lower().replace(" ", "_") for t, _ in sections[:20])
        db_utils.add_corpus_item(conn, source_id, "wiki_article", title, text,
                                 lang="en", meta={"url": url},
                                 tags=f"fandom {key} {tags}")

        # structured extraction
        summary = ""
        for t, x in sections:
            if t.lower().startswith(("personality", "background", "history")):
                summary = x[:1500]
                break
        summary = summary or text[:1200]

        if key in CHARACTER_PAGES:
            conn.execute(
                "INSERT OR REPLACE INTO characters(name, source_id, aliases_json, "
                "faction, role, summary, confidence, meta_json) VALUES (?,?,?,?,?,?,?,?)",
                (CHARACTER_PAGES[key], source_id,
                 json.dumps({k: v for k, v in ib.items()
                             if k in ("alias", "aliases", "species", "gender",
                                      "age", "vital_status")}, ensure_ascii=False),
                 ib.get("affiliation") or ib.get("occupation"),
                 "protagonist" if key == "fang_yuan" else "canon",
                 summary, "high", json.dumps(ib, ensure_ascii=False)))
        elif key in FACTION_PAGES:
            conn.execute(
                "INSERT OR REPLACE INTO factions(name, source_id, region, kind, "
                "summary, confidence) VALUES (?,?,?,?,?,?)",
                (FACTION_PAGES[key], source_id,
                 "Central Continent" if key == "heavenly_court" else "Southern Border",
                 "super_force" if key == "heavenly_court" else "clan",
                 summary, "high"))
        elif key in GU_PAGES:
            conn.execute(
                "INSERT OR REPLACE INTO gu_worms(name, source_id, rank, path, "
                "owner, summary, confidence) VALUES (?,?,?,?,?,?,?)",
                (GU_PAGES[key], source_id, ib.get("rank"), ib.get("path"),
                 ib.get("user") or ib.get("owner"), summary, "high"))
        conn.commit()
        n_files += 1
        print(f"[ok] {key}: {len(text)} chars, {len(sections)} sections, "
              f"infobox fields={len(ib)}")

    print(json.dumps(db_utils.stats(conn), indent=2))


if __name__ == "__main__":
    main()
