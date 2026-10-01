#!/usr/bin/env python3
"""Chapter-anchor scanner: ground canon entities in the actual novel corpus.

For each canon keyword group, find the chapter ranges where it appears most
densely, then:
1. store results in a `canon_anchors` table (created on the fly),
2. update events/decisions with anchor_chapters_json where matches exist,
3. report arc boundary evidence for the volume map.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import db_utils  # noqa: E402

ENTITY_GROUPS = {
    "Qing Mao Mountain": ["Qing Mao Mountain", "Qing Mao mountain"],
    "Flower Wine Temple": ["Flower Wine Temple"],
    "Liquor Worm": ["Liquor worm", "liquor worm"],
    "Spring Autumn Cicada": ["Spring Autumn Cicada", "Spring-Autumn Cicada"],
    "Bai Ning Bing": ["Bai Ning Bing"],
    "Fang Zheng": ["Fang Zheng"],
    "Gu Yue Bo": ["Gu Yue Bo"],
    "caravan journey": ["caravan"],
    "Shang clan city": ["Shang clan city", "Shang Clan City"],
    "Shang Yan Fei": ["Shang Yan Fei"],
    "Jia Jin Sheng": ["Jia Jin Sheng"],
    "Mudskin Toad": ["Mudskin toad", "Mudskin Toad"],
    "Imperial Court contest": ["Imperial Court contest", "Imperial Court"],
    "Chu Ying Ying": ["Chu Ying Ying"],
    "Hei Cheng": ["Hei Cheng"],
    "Tie Ruo Nan": ["Tie Ruo Nan"],
    "wolf tide": ["wolf tide"],
    "Heavenly Court": ["Heavenly Court"],
    "Longevity Heaven": ["Longevity Heaven"],
    "Crazed Demon Cave": ["Crazed Demon Cave"],
    "Reverse Flow River": ["Reverse Flow River", "reversed flow river"],
    "Three Kings": ["Three Kings"],
    "Yi Tian Mountain": ["Yi Tian Mountain"],
    "Xiao clan": ["Xiao clan"],
    "Blood Deity": ["Blood Deity", "blood deity"],
    "Spectral Soul": ["Spectral Soul"],
    "Thieving Heaven": ["Thieving Heaven"],
    "Giant Sun": ["Giant Sun"],
    "Star Constellation": ["Star Constellation"],
    "Duke Long": ["Duke Long"],
    "Feng Jin Huang": ["Feng Jin Huang"],
    "Zombie Dance": ["Zombie Dance"],
    "Blessed Land": ["blessed land"],
    "Immortal aperture": ["immortal aperture"],
    "dao marks": ["dao marks"],
    "Fixed Immortal Travel": ["Fixed Immortal Travel"],
    "Snowy Mountain Academy": ["Snowy Mountain"],
    "Gu formation": ["Gu formation"],
    " tribute": ["tribulation"],
}


def main() -> None:
    conn = db_utils.connect()
    db_utils.init_schema(conn)

    # locate novel chapters in corpus
    rows = conn.execute(
        "SELECT id, title, text, meta_json FROM corpus_items "
        "WHERE item_type='chapter'").fetchall()
    chapters = []
    for r in rows:
        meta = json.loads(r["meta_json"] or "{}")
        chapters.append((meta.get("chapter_id", 0), r["id"], r["title"], r["text"]))
    chapters.sort()
    print(f"scanning {len(chapters)} chapters...")

    conn.execute("""CREATE TABLE IF NOT EXISTS canon_anchors (
        id INTEGER PRIMARY KEY,
        entity TEXT NOT NULL,
        chapter_id INTEGER,
        corpus_id INTEGER,
        matches INTEGER,
        UNIQUE(entity, chapter_id))""")

    summary = {}
    for entity, variants in ENTITY_GROUPS.items():
        hits = []
        for ch_id, corpus_id, title, text in chapters:
            n = sum(text.count(v) for v in variants)
            if n > 0:
                hits.append((ch_id, corpus_id, n))
        hits.sort(key=lambda h: -h[2])
        for ch_id, corpus_id, n in hits[:40]:
            conn.execute(
                "INSERT OR REPLACE INTO canon_anchors(entity, chapter_id, "
                "corpus_id, matches) VALUES (?,?,?,?)",
                (entity, ch_id, corpus_id, n))
        conn.commit()
        if hits:
            chs = sorted(h[0] for h in hits)
            top = [h[0] for h in hits[:5]]
            summary[entity] = {"chapters": len(chs), "range": [chs[0], chs[-1]],
                               "top": top[:8], "total_matches": sum(h[2] for h in hits)}
    conn.commit()

    print(json.dumps(summary, indent=1)[:3000])

    # ---- update events with anchor chapters ----
    n_ev = 0
    for ev_id, title in conn.execute("SELECT id, title FROM events").fetchall():
        best = None
        for entity, s in summary.items():
            key = entity.lower().split()[0]
            if key and key in title.lower():
                best = s
                break
        if best:
            conn.execute(
                "UPDATE events SET chapter_range=? WHERE id=?",
                (json.dumps({"range": best["range"], "top": best["top"]}), ev_id))
            n_ev += 1
    conn.commit()
    print(f"[ok] events updated with chapter anchors: {n_ev}")


if __name__ == "__main__":
    main()
