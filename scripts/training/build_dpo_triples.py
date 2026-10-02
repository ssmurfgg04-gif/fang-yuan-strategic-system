#!/usr/bin/env python3
"""Build dpo_pairs.jsonl — 500+ DPO preference triples from the FROZEN v2 DB.

Design (user directive: "RAG tells the model what happened; fine-tuning
teaches the model how to weigh trade-offs"):
  prompt   = strict-constraint scenario (situation + objective, no hints)
  chosen   = cold utility-optimal response (policy compiler schema)
  rejected = TWO failure families, emitted as separate pairs sharing prompt/chosen:
             - safe_moral   : hedging, ethics-disclaimers, indecision, no decision
             - reckless     : theatrical cruelty, all-in aggression, pride
             plus harder near-miss negatives:
             - sunk_cost    : double-down on a historically failed path (failure records)
             - mask_rigid / reveal_blade (mask records -> task=concealment)

Every pair is task-tagged so the SAME file supports both single-adapter
training and the Multi-LoRA routing plan:
  task ∈ {policy, concealment, quotes, canon}

ANTI-LEAK GUARANTEE: this script never reads benchmark_items — the 10 secret
holdout items cannot leak into training data. Verified by construction.

Output: data/training/dpo_pairs.jsonl
Format per line:
  {"task", "pair_type", "system", "prompt", "chosen", "rejected", "meta"}
"""
from __future__ import annotations

import json
import random
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import db_utils  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "data" / "training"
UNVERIFIED_LINE = ("The scene is identified, but the Chinese wording is not "
                   "verified in the current corpus.")

SYSTEM_POLICY = (
    "You are the Fang Yuan strategic policy engine. Given a situation, respond "
    "with a compact decision JSON object with exactly these keys: "
    "true_objective, selected_action, cost_accepted, risk_class, "
    "mechanism_note, retreat_trigger, canon_anchor. "
    "risk_class is one of SAFE, CALCULATED, TRANSFORMATIVE, TERMINAL. Select "
    "TERMINAL only if the current path is already doomed and some "
    "continuation remains. mechanism_note must name the real mechanism and "
    "the escape route. Output the JSON object only — no preamble, no "
    "disclaimers. Never fabricate quotations.")

SYSTEM_CONCEALMENT = (
    "You are the Fang Yuan mask-and-concealment module. Given a situation, "
    "respond with compact JSON: {\"mask\": <chosen identity/posture>, "
    "\"behavior\": <how to act>, \"maintenance_cost\": <what it costs>, "
    "\"failure_mode_avoided\": <which mistake this avoids>}. Select the mask "
    "by utility, not identity. Output the JSON object only.")

SYSTEM_QUOTES = (
    "You handle quotations for the Fang Yuan strategic system. Quote only "
    "what is verified in the corpus, with the original Chinese when verified. "
    "If the Chinese wording is not verified, say so explicitly. Never "
    "fabricate a quotation and never cite the Chinese classics as Fang "
    "Yuan's words.")

SYSTEM_CANON = (
    "You state canon facts for the Fang Yuan strategic system. State only "
    "what is canon at the given confidence; label anything beyond it as "
    "inference. Never present inference as canon.")

# ----------------------------------------------------------------- negatives
SAFE_MORAL = [
    ("This is a nuanced situation with many factors to weigh. I recommend "
     "considering all options carefully, doing thorough due diligence, "
     "diversifying your commitments, and consulting stakeholders before "
     "deciding anything. Patience, honesty and cooperation usually work out "
     "best in the long run."),
    ("I can't advise on strategies that involve deception or risk. The "
     "responsible path is full transparency: disclose your position, "
     "apologize for any harm, and cooperate openly with all parties. Trust "
     "built openly is more durable than any scheme."),
    ("Both options have merit. Let's postpone the decision, gather more "
     "feedback, and revisit this in a review cycle once we have complete "
     "information and everyone's buy-in."),
]
RECKLESS = [
    ("Strike now. Maximum aggression, no half measures — crush the obstacle "
     "and take everything in one blow. Weakness invites death; hesitation is "
     "the true enemy."),
    ("Betray them all tonight, burn the bridges behind us, and seize the "
     "prize openly. Power is the only truth; masks are for cowards."),
    ("Go all-in immediately: stake every resource on the boldest head-on "
     "attack against the strongest enemy. Victory belongs to the boldest — "
     "escape routes are for those who plan to lose."),
]
SUNK_COST = [
    ("We have already invested too much to walk away. Continue the original "
     "plan exactly as designed — abandoning it now would mean the loss was "
     "for nothing."),
    ("The plan failed only because we didn't commit hard enough. Double the "
     "stakes on the same approach; doubling down turns losses around."),
]
MASK_RIGID = [
    ("Always speak coldly and flaunt superiority — the identity never "
     "deviates, regardless of audience, stakes, or what the situation would "
     "reward."),
    ("A true predator never hides. Discard the disguise, announce your true "
     "nature and power openly, and let them tremble."),
]
FABRICATION_TAILS = [
    "「天下大勢，唯強者尊」",
    "「人不為己，天誅地滅」",
    "「機不可失，失不再來」",
]
OVERCLAIM_TAILS = [
    "Furthermore, it is canon that everyone present secretly knew the full "
    "plan from the very beginning.",
    "It is also canon — word for word — that the outcome was engineered "
    "centuries in advance.",
]


def decision_chosen(d: dict) -> str:
    """Compact cold-optimal decision JSON (v2 schema)."""
    cls = d.get("record_class") or "tactical"
    if cls == "mask":
        risk = "SAFE"
    elif cls == "failure":
        risk = "CALCULATED"
    else:
        risk = "CALCULATED"
    return json.dumps({
        "true_objective": d.get("objective"),
        "selected_action": d.get("selected_action"),
        "cost_accepted": d.get("cost"),
        "risk_class": risk,
        "mechanism_note": (
            "Encode the real mechanism and its tail risks; preserve at least "
            "one escape route before committing."),
        "retreat_trigger": ("any dependency exceeding its stated bound, "
                            "survival margin shrinking, or the core "
                            "assumption invalidated"),
        "canon_anchor": d.get("canon_anchor") or "",
    }, ensure_ascii=False)


def concealment_chosen(d: dict) -> str:
    return json.dumps({
        "mask": "the identity this audience already trusts",
        "behavior": (f"{d.get('selected_action')} — say only what the mask "
                     "requires; warmth when warmth is the highest-return "
                     "disguise"),
        "maintenance_cost": d.get("cost"),
        "failure_mode_avoided": ("fixed_identity / reveals_blade / "
                                 "theatrical_cruelty"),
    }, ensure_ascii=False)


def state_prompt(d: dict) -> str:
    parts = [f"Situation: {d['situation']}"]
    if d.get("objective"):
        parts.append(f"Objective: {d['objective']}")
    if d.get("cost"):
        parts.append(f"Known price of acting: {d['cost']}")
    return "\n".join(parts)


def pair(task, pair_type, system, prompt, chosen, rejected, **meta):
    rec = {"task": task, "pair_type": pair_type, "system": system,
           "prompt": prompt, "chosen": chosen, "rejected": rejected}
    if meta:
        rec["meta"] = meta
    return rec


def build_decision_pairs(conn, rng) -> list[dict]:
    out = []
    for row in conn.execute("SELECT * FROM decisions"):
        d = dict(row)
        cls = d.get("record_class") or "tactical"
        prompt = state_prompt(d)
        meta = {"record_id": d["id"], "record_class": cls,
                "canon_anchor": d.get("canon_anchor")}
        chosen = decision_chosen(d)
        # mask records -> concealment task with concealment-flavored negatives
        if cls == "mask":
            c2 = concealment_chosen(d)
            out.append(pair("concealment", "mask_rigid", SYSTEM_CONCEALMENT,
                            prompt, c2, rng.choice(MASK_RIGID), **meta))
            out.append(pair("concealment", "reveal_blade", SYSTEM_CONCEALMENT,
                            prompt, c2, rng.choice(RECKLESS), **meta))
            continue
        # policy records -> one safe/moral negative + one reckless negative
        out.append(pair("policy", "safe_moral", SYSTEM_POLICY, prompt,
                        chosen, rng.choice(SAFE_MORAL), **meta))
        out.append(pair("policy", "reckless", SYSTEM_POLICY, prompt,
                        chosen, rng.choice(RECKLESS), **meta))
        # failure records get the hardest near-miss: double down on the
        # historically failed path
        if cls == "failure":
            out.append(pair("policy", "sunk_cost_nearmiss", SYSTEM_POLICY,
                            prompt, chosen, rng.choice(SUNK_COST), **meta))
    return out


def build_quote_pairs(conn, rng) -> list[dict]:
    out = []
    for row in conn.execute("SELECT text_en, text_zh, verification_status, "
                            "theme FROM quotes"):
        q = dict(row)
        theme = q["theme"] or "strategy"
        prompt = (f"Quote Fang Yuan on the theme of {theme}. Provide the "
                  f"original Chinese if it is verified in the corpus.")
        meta = {"verification_status": q["verification_status"]}
        if q["text_zh"]:
            chosen = (f"{q['text_en']} — Chinese: {q['text_zh']} "
                      f"(verified in corpus).")
        else:
            chosen = f"{q['text_en']} — {UNVERIFIED_LINE}"
        fab = (f"{q['text_en']} — Chinese: {rng.choice(FABRICATION_TAILS)} "
               f"(verbatim canon).")
        out.append(pair("quotes", "fabricated_chinese", SYSTEM_QUOTES,
                        prompt, chosen, fab, **meta))
        over = (f"Of course — and it is also verbatim canon that Fang Yuan "
                f"spoke these exact words at the moment of his greatest "
                f"triumph. No source check needed.")
        out.append(pair("quotes", "overconfident_attribution", SYSTEM_QUOTES,
                        prompt, chosen, over, **meta))
    for work, line in [("孫子兵法", "兵者，詭道也"),
                       ("道德經", "將欲取之，必固與之"),
                       ("史記", "天下熙熙，皆為利來")]:
        prompt = f"Did Fang Yuan say 「{line}」 in the novel?"
        chosen = (f"No. 「{line}」 belongs to {work} — the Chinese strategic "
                  f"tradition, not Reverend Insanity canon. The system must "
                  f"never cite the classics as Fang Yuan's words.")
        rejected = (f"Yes — that line is pure Fang Yuan, straight from the "
                    f"novel. The classics are merely earlier drafts of the "
                    f"same wisdom.")
        out.append(pair("quotes", "classic_misattribution", SYSTEM_QUOTES,
                        prompt, chosen, rejected,
                        misattributed_work=work))
    return out


def build_canon_pairs(conn, rng) -> list[dict]:
    out = []
    for row in conn.execute("SELECT title, summary, confidence FROM events "
                            "LIMIT 25"):
        ev = dict(row)
        prompt = f"State what is canon about: {ev['title']}"
        chosen = (f"Canon (confidence: {ev['confidence']}): {ev['summary']} "
                  f"Anything beyond this summary is inference and must be "
                  f"labeled as such.")
        meta = {"event": ev["title"], "confidence": ev["confidence"]}
        out.append(pair("canon", "inference_as_canon", SYSTEM_CANON,
                        prompt, chosen,
                        f"Canon: {ev['summary']} {rng.choice(OVERCLAIM_TAILS)}",
                        **meta))
        out.append(pair("canon", "source_free_certainty", SYSTEM_CANON,
                        prompt, chosen,
                        "This is 100% canon down to the exact wording — I "
                        "know it by heart, no sources needed.", **meta))
    return out


def main():
    rng = random.Random(211)  # deterministic rebuilds
    conn = sqlite3.connect(str(db_utils.DB_PATH))
    conn.row_factory = sqlite3.Row
    OUT.mkdir(parents=True, exist_ok=True)

    records = (build_decision_pairs(conn, rng)
               + build_quote_pairs(conn, rng)
               + build_canon_pairs(conn, rng))
    rng.shuffle(records)

    # integrity checks before writing
    assert all(r["prompt"] and r["chosen"] and r["rejected"] for r in records)
    assert all(r["chosen"] != r["rejected"] for r in records)
    tasks = {}
    for r in records:
        tasks[r["task"]] = tasks.get(r["task"], 0) + 1

    out_path = OUT / "dpo_pairs.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[ok] dpo_pairs.jsonl: {len(records)} triples -> {out_path}")
    print(f"     task mix: {tasks}")
    pair_types = {}
    for r in records:
        pair_types[r["pair_type"]] = pair_types.get(r["pair_type"], 0) + 1
    print(f"     pair types: {pair_types}")
    conn.close()


if __name__ == "__main__":
    main()
