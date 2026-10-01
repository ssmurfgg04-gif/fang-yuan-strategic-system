#!/usr/bin/env python3
"""Load canon seed data (events, decisions, quotes, characters, factions, gu,
locations) into the master DB with provenance = analyst_generated."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import db_utils  # noqa: E402

SEED = Path(__file__).resolve().parent / "data"


def load_json(name: str) -> dict | list:
    return json.loads((SEED / name).read_text(encoding="utf-8"))


def main() -> None:
    conn = db_utils.connect()
    db_utils.init_schema(conn)

    sid = db_utils.ensure_source(
        conn, source_key="analyst:seed_canon",
        kind="analyst_generated",
        title="Canon knowledge seeds (analyst-authored, fandom-cross-checkable)",
        license_="Original analysis; anchors to CC BY-SA wiki and public novel knowledge",
        provenance_status="analyst_generated",
        meta={"note": "Confidence flags mark what needs verification against "
                      "fandom wiki or user-provided corpus"})

    # events
    for ev in load_json("events_seed.json")["events"]:
        exists = conn.execute("SELECT 1 FROM events WHERE title=?",
                              (ev["title"],)).fetchone()
        if exists:
            continue
        conn.execute(
            "INSERT INTO events(source_id, title, arc, volume, summary, "
            "participants_json, confidence) VALUES (?,?,?,?,?,?,?)",
            (sid, ev["title"], ev["arc"], ev.get("volume"), ev["summary"],
             json.dumps(ev.get("participants", []), ensure_ascii=False),
             ev.get("confidence", "medium")))
    conn.commit()

    # decisions (3 parts)
    n_dec = 0
    for part in ("decisions_seed_part1.json", "decisions_seed_part2.json",
                 "decisions_seed_part3.json", "decisions_seed_part4.json",
                 "decisions_seed_part5.json", "decisions_seed_part6.json"):
        for d in load_json(part)["decisions"]:
            exists = conn.execute(
                "SELECT 1 FROM decisions WHERE situation=? AND selected_action=?",
                (d["situation"], d["selected_action"])).fetchone()
            if exists:
                continue
            db_utils.add_decision(
                conn, source_id=sid, record_class=d["record_class"],
                situation=d["situation"], objective=d.get("objective"),
                known_facts=d.get("known_facts"), unknowns=d.get("unknowns"),
                resources=d.get("resources"), options=d.get("options"),
                selected_action=d["selected_action"],
                rejected_options=d.get("rejected_options"),
                cost=d.get("cost"), result=d.get("result"),
                lesson=d.get("lesson"), canon_anchor=d.get("canon_anchor"),
                volume=d.get("volume"), confidence=d.get("confidence", "medium"))
            n_dec += 1

    # quotes
    n_q = 0
    for q in load_json("quotes_seed.json")["quotes"]:
        exists = conn.execute("SELECT 1 FROM quotes WHERE text_en=?",
                              (q["text_en"],)).fetchone()
        if exists:
            continue
        db_utils.add_quote(
            conn, source_id=sid, text_en=q["text_en"], text_zh=q.get("text_zh"),
            pinyin=q.get("pinyin"), translation_kind=q.get("translation_kind"),
            speaker=q["speaker"], chapter_ref=q.get("chapter_ref"),
            scene=q["scene"], theme=q["theme"],
            verification_status=q["verification_status"],
            confidence=q.get("confidence", "low"))
        n_q += 1

    # extra structured canon: locations + additional characters/factions/gu
    locations = [
        ("Qing Mao Mountain", "Southern Border", "Fang Yuan's home mountain; Gu Yue clan village; site of awakening, Liquor Worm scheme, and the escape.", "high"),
        ("Flower Wine Temple", "Southern Border", "Legacy site on Qing Mao Mountain holding the Liquor Worm.", "high"),
        ("Shang clan city", "Southern Border", "Major commercial hub of the Shang clan; commerce and mask-management arc.", "medium"),
        ("Northern Plains", "Northern Plains", "Tribal grassland region; Imperial Court contest and tribal politics.", "high"),
        ("Central Continent", "Central Continent", "Heart of the Gu world; home of Heavenly Court and ten great sects.", "high"),
        ("Western Desert", "Western Desert", "Region of the Xiao clan spider trade and desert politics.", "medium"),
        ("Eastern Sea", "Eastern Sea", "Sea-path region; powerful seafaring forces.", "medium"),
        ("Reverse Flow River", "n/a", "River of Time mechanics site; world-scale strategic period.", "medium"),
        ("Crazed Demon Cave", "n/a", "Rule-based pocket world; deduction and Dao-mark battlefield.", "medium"),
        ("Imperial Court", "Northern Plains", "Tribal contest arena for supreme position in Northern Plains politics.", "low"),
        ("Longevity Heaven", "Northern Plains", "Super-force heaven of the Northern Plains, counterpart to Heavenly Court.", "medium"),
        ("Three Kings Blessed Land", "n/a", "Inheritance-type blessed land asset referenced in mid-novel arcs.", "low"),
    ]
    for name, region, summary, conf in locations:
        exists = conn.execute("SELECT 1 FROM locations WHERE name=?", (name,)).fetchone()
        if not exists:
            conn.execute(
                "INSERT INTO locations(name, region, summary, confidence) VALUES (?,?,?,?)",
                (name, region, summary, conf))
    conn.commit()

    extras = {
        "characters": [
            ("Gu Yue Bo", "Gu Yue Clan", "Southern Border", "clan leader / antagonist-adjacent", "Clan leader of Gu Yue village; manages the response to Fang Yuan's rise.", "high"),
            ("Gu Yue Mo Bei", "Gu Yue Clan", "Southern Border", "law enforcement", "Clan enforcement elder who watches Fang Yuan's anomalies.", "low"),
            ("Chu Ying Ying", "Northern Plains tribes", "Northern Plains", "ally / merchant genius", "Merchant-genius ally during the Northern Plains period.", "medium"),
            ("Hei Cheng", "Northern Plains tribes", "Northern Plains", "ally / warrior", "Warrior ally in the Northern Plains period.", "medium"),
            ("Tie Ruo Nan", "Tie clan", "Northern Plains", "righteous pursuer", "Righteous-path pursuer associated with the Imperial Court contest period.", "low"),
            ("Jia Jin Sheng", "merchant circle", "Southern Border", "merchant counterparty", "Merchant whose transaction taught the shadow-cost lesson.", "medium"),
        ],
        "factions": [
            ("Xiao clan", "Western Desert", "clan", "Clan whose transport/economic formation's recurring input Fang Yuan supplied (spider trade).", "medium"),
            ("Ba clan", "Southern Border", "clan", "Merchant-clan associated with the caravan journey period.", "low"),
            ("Longevity Heaven", "Northern Plains", "super_force", "Northern Plains super-force; counterpart to Heavenly Court.", "medium"),
            ("Shadow Sect", "n/a", "sect", "Covert sect manipulating events across regions in mid-late novel.", "low"),
            ("Ten Venerables (institution)", "n/a", "institution", "The ten peak Sovereigns/Venerables of Gu world history whose legacies structure the endgame.", "medium"),
        ],
        "gu_worms": [
            ("Moonlight Gem", "Rank 1", "moon", "Fang Yuan", "Basic moon-path attack/light Gu common in Volume 1.", "high"),
            ("Yin-Yang Rotation Gu", "unknown", "unknown", "Bai Ning Bing (associated)", "Escape-enabling Gu referenced in the Qing Mao Mountain escape.", "low"),
            ("Mudskin Toad", "unknown", "body/concealment", "Fang Yuan", "Concealment/survival Gu from the caravan period.", "low"),
            ("Fixed Immortal Travel", "Immortal Gu", "space", "Fang Yuan (later)", "Spatial escape Immortal Gu associated with Fang Yuan's mid-novel mobility.", "low"),
            ("Myriad Self", "Immortal Gu", "transformation", "Fang Yuan (later)", "Transformation-path Immortal Gu associated with late-novel Fang Yuan.", "low"),
        ],
    }
    for table, rows in extras.items():
        for row in rows:
            if table == "characters":
                exists = conn.execute("SELECT 1 FROM characters WHERE name=?", (row[0],)).fetchone()
                if not exists:
                    conn.execute(
                        "INSERT INTO characters(source_id, name, faction, region, role, "
                        "summary, confidence) VALUES (?,?,?,?,?,?,?)",
                        (sid, row[0], row[1], row[2], row[3], row[4], row[5]))
            elif table == "factions":
                exists = conn.execute("SELECT 1 FROM factions WHERE name=?", (row[0],)).fetchone()
                if not exists:
                    conn.execute(
                        "INSERT INTO factions(source_id, name, region, kind, summary, confidence) "
                        "VALUES (?,?,?,?,?,?)", (sid, row[0], row[1], row[2], row[3], row[4]))
            elif table == "gu_worms":
                exists = conn.execute("SELECT 1 FROM gu_worms WHERE name=?", (row[0],)).fetchone()
                if not exists:
                    conn.execute(
                        "INSERT INTO gu_worms(source_id, name, rank, path, owner, summary, confidence) "
                        "VALUES (?,?,?,?,?,?,?)", (sid, row[0], row[1], row[2], row[3], row[4], row[5]))
    conn.commit()

    print(f"[ok] decisions inserted this run: {n_dec}")
    print(f"[ok] quotes inserted this run: {n_q}")
    print(json.dumps(db_utils.stats(conn), indent=2))


if __name__ == "__main__":
    main()
