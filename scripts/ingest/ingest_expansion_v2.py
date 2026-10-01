#!/usr/bin/env python3
"""Expansion v2: lessons (GreenBamboo-v1 analysis + v2 engineering) and
findings (research round 2)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import db_utils  # noqa: E402

LESSONS = [
    # --- GreenBamboo-v1 study (surpass-the-previous-approach directive) ---
    {"phase": "training", "category": "underweighted", "statement": "GreenBamboo-v1 (12B Gemma-3 finetune) proved ideology/tone finetuning works but explicitly cannot recall canon plot.", "detail": "Their design goal was a roleplay GM: ideology, tone, mechanics — not a plot database. Community confirmed the gap ('use a wiki'). Our system answers it with structured RAG + provenance: the model stays small and policy-shaped while retrieval carries canon facts with chapter citations.", "origin": "greenbamboo_study"},
    {"phase": "training", "category": "engineering", "statement": "What-if scenario datasets are the core teaching instrument (GreenBamboo's key technique).", "detail": "Their dataset was built from counterfactual scenarios ('what if Fang Yuan relied on someone? what if he used more politics?'). We adopt and exceed: our counterfactual benchmark layer (with graded variants) doubles as a training-data generator — variant families with graded rubrics become SFT/DPO pairs automatically.", "origin": "greenbamboo_study"},
    {"phase": "training", "category": "failure_mode", "statement": "Creative finetunes stack badly: finetuning on an already-uncensored creative base produces a 'dumber version'.", "detail": "GreenBamboo author's own observation. Implication: our Stage-A LoRA should start from the pristine instruct base, never from roleplay/uncensored community merges — even when they promise easier compliance.", "origin": "greenbamboo_study"},
    {"phase": "ops", "category": "engineering", "statement": "Chat-template sensitivity: Gemma-3 series degrades badly with the wrong template.", "detail": "A user-facing failure of GreenBamboo-v1 was traced to template mismatch. For our Qwen2.5 path: always ship the exact chat template with the adapter, validate template integrity in the eval harness, and fail loudly on template mismatch instead of silently degrading.", "origin": "greenbamboo_study"},
    {"phase": "ops", "category": "failure_mode", "statement": "Length-instruction compliance fails on small finetunes trained for short turns.", "detail": "GreenBamboo users found it could not produce long outputs (trained for back-and-forth GM turns). Our design embraces this: the policy actor is capped at 80–180 token JSON decisions; long-form text is a formatter's job, separable from the policy.", "origin": "greenbamboo_study"},
    {"phase": "ops", "category": "engineering", "statement": "RAG pairing is the community's first request for lore finetunes — we build it in from day one.", "detail": "Top comment on the GreenBamboo thread: 'RAG would be quite useful actually instead of a wiki.' Our DB+FTS5 retrieval with chapter citations is that pairing, with provenance and verification the wiki lacks.", "origin": "greenbamboo_study"},
    {"phase": "training", "category": "engineering", "statement": "GM-mode and policy-mode are different products; do not blur them.", "detail": "GreenBamboo serves GM roleplay (world-flavored narration). Our target is decision fidelity (would Fang Yuan choose this?). The formatter supports both modes but the training label vector (attachment_free, objective_fixed...) is a policy label, not a narration label.", "origin": "greenbamboo_study"},
    {"phase": "training", "category": "engineering", "statement": "Model-size tradeoff: 12B gives fluency for roleplay; 0.5B gives controllability for policy.", "detail": "GreenBamboo chose 12B Gemma-3 (Q8_0/Q5_K_M GGUF) for performance tone. We chose 0.5B for surgical policy control with external systems (DB/simulator/search) carrying the load — smaller bases bring weaker assistant habits to overwrite.", "origin": "greenbamboo_study"},
    # --- v2 engineering session ---
    {"phase": "corpus", "category": "engineering", "statement": "Merging the user's cleaned novel corpus (2063 chapters, 4.28M words) upgrades the whole system to chapter-anchored grounding.", "detail": "Entity-density scanning (anchor_scan.py) grounds canon entities in real chapter ranges: Shang city 229-396, Qing Mao ends ~191, Jia Jin Sheng 42-236, Crazed Demon Cave 1173-2048, Heavenly Court 1005+. Events and decisions now cite measured chapter evidence instead of arc guesses.", "origin": "v2_session"},
    {"phase": "corpus", "category": "engineering", "statement": "Copyright boundary for the merged corpus: local master DB holds novel text; the public repo receives a body-stripped variant.", "detail": "The user-provided corpus is provenance=user_responsibility, kept in data/inbox (gitignored) and in the local/download DB. The GitHub push carries fang_yuan_public.db with chapter index/titles/translator metadata only — no body text leaves the user's control via this project.", "origin": "v2_session"},
    {"phase": "policy", "category": "engineering", "statement": "Alliances sell restraint; enemies' fear is a product with multiple markets.", "detail": "Non-aggression annuities can be quietly priced to rival buyers of the same fear. Recorded as decision doctrine (transaction class).", "origin": "v2_session"},
    {"phase": "policy", "category": "engineering", "statement": "Discovered spies are infrastructure: the handler is the target, the spy is the socket.", "detail": "Invert rather than punish: controlled information flows while recruiting upward.", "origin": "v2_session"},
    {"phase": "benchmark", "category": "engineering", "statement": "Benchmark integrity note: the mock policy oracle reads the policy spec for counterfactual postures — it calibrates the grader, not model capability.", "detail": "Real capability measurement targets LLM-backed adapters; persona baselines answer honestly from their scripts.", "origin": "v2_session"},
    {"phase": "ops", "category": "engineering", "statement": "Stage decisions by irreversibility when information velocity outruns verification.", "detail": "Tiered flow: auto-act small reversible matters on source grade; hedge partial actions on medium; full verification only for irreversible. (Decision doctrine, tactical class.)", "origin": "v2_session"},
    {"phase": "corpus", "category": "legal", "statement": "Zizhi Tongjian full text is not available as plain text on zh.wikisource (四部叢刊本 pages are scan containers).", "detail": "Documented gap; candidates: zh.wikisource other editions, ctext.org (respecting no-scraping), or paid licensed source. The 才德论 principles were captured via v1 authored summary as placeholder.", "origin": "v2_session"},
    {"phase": "corpus", "category": "engineering", "statement": "Classics expansion completed 137 units: Han Feizi deepened to 17 pian (incl. 說林 case collections), Shiji biographies added (項羽/高祖/留侯/淮陰侯/商君/刺客/李將軍/老子韓非), Zhanguoce six 策, Zizhi Tongjian placeholder, Shangjunshu four 卷.", "detail": "Every unit carries analyst summary + principles + Fang Yuan anchor; 說林上/下 directly validate the decision-record case-library design.", "origin": "v2_session"},
    {"phase": "training", "category": "engineering", "statement": "211 decision records now span all five record classes with chapter-anchored evidence where the scan could confirm.", "detail": "Distribution: 68 tactical, 40 mask, 35 failure, 35 major, 33 transaction. Failure records are the highest-value class for policy surgery — they teach the model what NOT to repeat.", "origin": "v2_session"},
    {"phase": "ops", "category": "engineering", "statement": "Concession discipline: concessions need price tags and cliffs; stop escalation with demonstrations, not gifts.", "detail": "Face-saving gifts read as floors in some negotiation cultures. (Failure-class record.)", "origin": "v2_session"},
    {"phase": "ops", "category": "engineering", "statement": "Diversification is a graph property, not a count: correlated option legs fail together.", "detail": "The scoring model must dependency-graph its legs and cap exposure per underlying node — mirrors simulator finding F-2 (conditionality pricing).", "origin": "v2_session"},
]

FINDINGS = [
    {"topic": "greenbamboo", "finding": "GreenBamboo-v1: 12B Gemma-3 finetune, GGUF Q8_0/Q5_K_M, roleplay-GM design; strong ideology/tone, explicitly weak canon recall (author: 'not a plot database'). Trained on what-if scenario datasets.", "confidence": "high"},
    {"topic": "greenbamboo", "finding": "Community top request: pair the finetune with RAG instead of a wiki — validates our retrieval-first architecture.", "confidence": "high"},
    {"topic": "greenbamboo", "finding": "Failure reports: length-instruction non-compliance (trained for short GM turns), chat-template sensitivity (Gemma-3), stacked-creative-finetune degradation.", "confidence": "high"},
    {"topic": "canon_structure", "finding": "Chapter-anchored entity scan (2063 chapters, 4.28M words): Qing Mao Mountain arc ends ~ch191 (Gu Yue Bo last mention 191); Shang clan city 229-396; Jia Jin Sheng 42-236; wolf tide mentions 93-509; Imperial Court contest 478-1979; Hei Cheng 528-1125; Snowy Mountain Academy 537-1282; Heavenly Court 1005-2031; Reverse Flow River 1010-1946; Crazed Demon Cave 1173-2048.", "confidence": "high"},
    {"topic": "canon_structure", "finding": "Volume pages on fandom are link stubs; measured entity density is a better arc-boundary source than scraped volume lists.", "confidence": "high"},
    {"topic": "classics", "finding": "Expansion sources verified on zh.wikisource: Hanfeizi 8 more pian (八姦/有度/十過/備內/亡徵/六反/顯學/南面/大體/觀行/用人/外儲說左上/說林上下), Shiji biographies (卷063/065/068/086/092/007/055/008/109), Zhanguoce (齊一/齊四/楚一/趙一/魏二/燕一/秦三), Shangjunshu 卷三/卷四 — all pre-1912 public domain.", "confidence": "high"},
    {"topic": "classics", "finding": "資治通鑑 (四部叢刊本) pages are scan containers (~150 chars wikitext each); prop=extracts returns empty. Full-text needs ctext.org (respect no-scraping) or licensed source.", "confidence": "high"},
    {"topic": "engineering", "finding": "Chapter-anchored decision records: canon_anchor strings now cross-reference measured chapter ranges from the anchor scan, enabling retrieval-grounded canon answers with chN citations.", "confidence": "high"},
    {"topic": "engineering", "finding": "DB merge: user's reverend_insanity.db (2063 chapters + 311 notes, per-chapter sha256 + translator metadata) merged into master DB as corpus_items with provenance=user_responsibility; FTS alignment maintained via explicit rowid insertion.", "confidence": "high"},
    {"topic": "policy", "finding": "New decision doctrines added from corpus study: non-aggression annuities sold to multiple fear-buyers; spy inversion (handler as target); evacuation-before-strike for hostage shields; signature-seasoning against counter-treasures; staged tribulation decoys; jurisdictional-seam endgame extraction.", "confidence": "medium"},
]

def main():
    conn = db_utils.connect()
    db_utils.init_schema(conn)
    n_lessons = n_findings = 0
    for L in LESSONS:
        exists = conn.execute("SELECT 1 FROM lessons WHERE statement=?", (L["statement"],)).fetchone()
        if not exists:
            db_utils.add_lesson(conn, L["phase"], L["category"], L["statement"],
                                L.get("detail"), L.get("origin"))
            n_lessons += 1
    for F in FINDINGS:
        exists = conn.execute("SELECT 1 FROM findings WHERE finding=?", (F["finding"],)).fetchone()
        if not exists:
            db_utils.add_finding(conn, F["topic"], "research round 2", "",
                                 "research", F["finding"], F.get("confidence", "medium"))
            n_findings += 1
    print(f"[ok] lessons +{n_lessons} (total {conn.execute('SELECT COUNT(*) FROM lessons').fetchone()[0]})")
    print(f"[ok] findings +{n_findings} (total {conn.execute('SELECT COUNT(*) FROM findings').fetchone()[0]})")

    # ingest the r2 search results as findings too
    import glob
    n2 = 0
    for f in glob.glob(str(Path(__file__).resolve().parents[2] / "data" / "raw" / "search_r2_*.json")):
        try:
            items = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        for it in items:
            snip = (it.get("snippet") or "").strip()
            if not snip:
                continue
            exists = conn.execute("SELECT 1 FROM findings WHERE source_url=?", (it.get("url",""),)).fetchone()
            if exists:
                continue
            db_utils.add_finding(conn, "research_r2", "", it.get("url",""), it.get("name",""), snip)
            n2 += 1
    print(f"[ok] r2 search findings +{n2}")


if __name__ == "__main__":
    main()
