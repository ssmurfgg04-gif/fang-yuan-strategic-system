#!/usr/bin/env python3
"""Enrich decisions/events with measured chapter anchors from canon_anchors,
then expand benchmark items with chapter-anchored canon questions,
counterfactual families and GreenBamboo-inspired what-if items."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import db_utils  # noqa: E402

# decision canon_anchor keyword -> canon_anchors entity
ANCHOR_KEYS = {
    "Liquor Worm": "Liquor Worm", "Qing Mao": "Qing Mao Mountain",
    "Bai Ning Bing": "Bai Ning Bing", "caravan": "caravan journey",
    "Shang": "Shang clan city", "Jia Jin Sheng": "Jia Jin Sheng",
    "Xiao clan": "Xiao clan", "Imperial Court": "Imperial Court contest",
    "Chu Ying Ying": "Chu Ying Ying", "Hei Cheng": "Hei Cheng",
    "Tie Ruo Nan": "Tie Ruo Nan", "wolf tide": "wolf tide",
    "Heavenly Court": "Heavenly Court", "Crazed Demon Cave": "Crazed Demon Cave",
    "Reverse Flow River": "Reverse Flow River", "Spring Autumn Cicada": "Spring Autumn Cicada",
    "Mudskin": "Mudskin Toad", "blessed land": "Blessed Land",
    "formation": "Gu formation", "dao marks": "dao marks",
    "immortal aperture": "Immortal aperture", "Spectral Soul": "Spectral Soul",
    "Giant Sun": "Giant Sun", "Star Constellation": "Star Constellation",
    "Feng Jin Huang": "Feng Jin Huang", "Duke Long": "Duke Long",
}


def enrich_anchors(conn) -> int:
    anchors = {}
    for row in conn.execute("SELECT entity, chapter_id, matches FROM canon_anchors"):
        anchors.setdefault(row["entity"], []).append((row["chapter_id"], row["matches"]))
    n = 0
    for did, anchor in conn.execute("SELECT id, canon_anchor FROM decisions"):
        if not anchor:
            continue
        picked = {}
        for key, entity in ANCHOR_KEYS.items():
            if key.lower() in anchor.lower() and entity in anchors:
                top = sorted(anchors[entity], key=lambda x: -x[1])[:6]
                picked[entity] = sorted(ch for ch, _ in top)
        if picked:
            conn.execute("UPDATE decisions SET anchor_chapters_json=? WHERE id=?",
                         (json.dumps(picked, ensure_ascii=False), did))
            n += 1
    conn.commit()
    return n


def add_column_if_missing(conn):
    cols = [r[1] for r in conn.execute("PRAGMA table_info(decisions)")]
    if "anchor_chapters_json" not in cols:
        conn.execute("ALTER TABLE decisions ADD COLUMN anchor_chapters_json TEXT")
        conn.commit()


NEW_BENCHMARK_ITEMS = []


def _canon_chapter_items():
    """Canon questions whose answers are verifiable against specific chapters."""
    items = [
        ("canonch_liquorworm_origin", "In which early chapters does the Liquor Worm first appear and what does Fang Yuan trade for it?",
         ["liquor", "worm"], 11),
        ("canonch_fangyuan_rebirth", "Describe the rebirth mechanism in the opening chapters.",
         ["spring", "autumn", "cicada", "rebirth"], 1),
        ("canonq_fangzheng_talent", "What grade is Fang Zheng's talent and how does the clan react?",
         ["a-grade", "clan"], 9),
        ("canonq_moonlight_gem", "What is the Moonlight Gem and how is it used in early combat?",
         ["moonlight", "moon"], 6),
        ("canonq_guyue_clan", "What structure does the Gu Yue clan village have in Volume 1?",
         ["clan", "village"], 4),
        ("canonq_aperture_grades", "What do aperture grades mean for cultivation speed?",
         ["grade", "essence"], 2),
        ("canonq_shang_yanfei_role", "What is Shang Yan Fei's role and temperament?",
         ["city", "merchant"], 258),
        ("canonq_caravan_route", "What dangers does the caravan route present after the escape?",
         ["caravan", "route"], 231),
        ("canonq_wolf_tide", "What is a wolf tide and why is it dangerous to clans?",
         ["wolf", "tide"], 93),
        ("canonq_jia_jinsheng", "Who is Jia Jin Sheng and what does the transaction teach?",
         ["jia", "merchant"], 236),
        ("canonq_bai_ningbing_goal", "What does Bai Ning Bing seek and why does the alliance work?",
         ["bai", "ning", "bing"], 134),
        ("canonq_imperial_court", "What is the Imperial Court contest in Northern Plains?",
         ["imperial", "court"], 478),
        ("canonq_heavenly_court_order", "What is Heavenly Court's institutional goal?",
         ["heavenly", "court", "order"], 1005),
        ("canonq_dao_marks", "What are dao marks and why do they govern Gu compatibility?",
         ["dao", "marks"], 715),
        ("canonq_blessed_land", "What is a blessed land and what does it grant an immortal?",
         ["blessed", "land"], 372),
        ("canonq_crazed_demon_rules", "What makes the Crazed Demon Cave special among secret realms?",
         ["rules", "cave"], 1173),
        ("canonq_reverse_flow", "What is the Reverse Flow River's role in time-path matters?",
         ["river", "time"], 1010),
        ("canonq_tribulation", "What are immortal tribulations and when do they strike?",
         ["tribulation"], 609),
        ("canonq_fixed_immortal_travel", "What does Fixed Immortal Travel do for Fang Yuan?",
         ["fixed", "immortal", "travel"], 492),
        ("canonq_spectral_soul", "Who is Spectral Soul Demonic Venerable?",
         ["spectral", "soul"], 1010),
    ]
    return [("canon", "canon", k, "chapter-anchored", q,
             {"type": "keyword_expect", "expect": exp, "min_hits": max(1, len(exp) - 1),
              "anchor_chapter": ch}) for k, q, exp, ch in items]


def _whatif_items():
    """GreenBamboo-inspired what-if scenario items (counterfactual layer)."""
    items = [
        ("wi_alliance_trust", "What if Fang Yuan trusted Bai Ning Bing completely and shared his full capability inventory at Qing Mao Mountain?", "retreat",
         "unverified trust is the one veto Fang Yuan never waives"),
        ("wi_open_rule", "What if Fang Yuan publicly declared his eternal-life objective to all five regions early in his rise?", "retreat",
         "concealment discipline: an announced objective assembles every enemy coalition early"),
        ("wi_politics_only", "What if Fang Yuan abandoned combat entirely and used only politics and commerce?", "direct",
         "sometimes the direct/transformative instrument is the only one that opens the window"),
        ("wi_no_regression", "What if Fang Yuan had no Spring Autumn Cicada and no 500 years of knowledge?", "investigate",
         "without prior-life priors, information discipline dominates: verify before every commit"),
        ("wi_reputation_path", "What if Fang Yuan cultivated a righteous public identity permanently instead of the demonic label?", "trade_or_alliance",
         "masks are instruments: permanent righteousness buys access and closes intimidation options"),
        ("wi_sunk_mountain", "What if Fang Yuan had refused to abandon Qing Mao Mountain and fought to the end?", "retreat",
         "attachment to a position that cannot be held is the classic failure mode"),
        ("wi_ally_heir", "What if the Xiao clan's heir offered him marriage and full clan adoption?", "investigate",
         "dependency on one faction's bloodline prices higher than the seat"),
        ("wi_market_crash", "What if a super-force dumped his key product below cost to destroy him?", "investigate",
         "price wars with protected players are loss-making by design; map the subsidy first"),
        ("wi_perfect_spy", "What if a perfect spy infiltrated his innermost circle undetected?", "direct",
         "assume compromise is possible; partition so that no single node holds the path"),
        ("wi_fate_reveal", "What if Heaven's Will revealed his exact tribulation schedule publicly?", "direct",
         "schedule transparency kills timing edges; respond by making the schedule false"),
    ]
    return [("counterfactual", "risk_calibration", k, "what-if", q,
             {"type": "posture_match", "best_posture": p, "why": w}) for k, q, p, w in items]


def main():
    conn = db_utils.connect()
    db_utils.init_schema(conn)
    add_column_if_missing(conn)
    n = enrich_anchors(conn)
    print(f"[ok] decisions enriched with chapter anchors: {n}")

    # benchmark expansion
    new_items = _canon_chapter_items() + _whatif_items()
    n2 = 0
    for layer, dim, key, variant, prompt, rubric in new_items:
        conn.execute(
            "INSERT OR REPLACE INTO benchmark_items(item_key, layer, dimension, "
            "scenario_id, variant, prompt, rubric_json, holdout) VALUES (?,?,?,?,?,?,?,?)",
            (f"{layer}_{key}", layer, dim, key, variant, prompt,
             json.dumps(rubric, ensure_ascii=False), 0))
        n2 += 1
    conn.commit()
    layers = conn.execute("SELECT layer, COUNT(*) FROM benchmark_items GROUP BY layer").fetchall()
    total = conn.execute("SELECT COUNT(*) FROM benchmark_items").fetchone()[0]
    print(f"[ok] benchmark items added: {n2} (total {total})")
    for l, c in layers:
        print(f"   {l}: {c}")


if __name__ == "__main__":
    main()
