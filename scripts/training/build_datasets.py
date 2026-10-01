#!/usr/bin/env python3
"""Training dataset builders for Qwen2.5-0.5B policy surgery.

Outputs (JSONL under data/training/):
- sft_decisions.jsonl     : Stage-A supervised policy tuning records
- preference_pairs.jsonl  : Stage-B chosen/rejected pairs
- fake_fang_yuan.jsonl    : negative examples with failure labels + replacements
- quote_verification.jsonl: verify-don't-fabricate behavior
- canon_vs_inference.jsonl: provenance discipline records
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

SYSTEM_PROMPT = (
    "You are the Fang Yuan strategic policy engine. Given a situation, output "
    "a compact decision JSON: true_objective, decisive_resource, "
    "highest_value_unknown, options (each with action/gain/cost/risk_class/"
    "escape_route), selected_action, why_now, retreat_trigger, canon_anchor. "
    "Risk classes: SAFE, CALCULATED, TRANSFORMATIVE, TERMINAL. Never select "
    "TERMINAL risk unless the current path is already doomed. Never fabricate "
    "quotations.")


def state_block(d: dict) -> str:
    facts = "\n".join(f"- {x}" for x in d.get("known_facts") or [])
    unknowns = "\n".join(f"- {x}" for x in d.get("unknowns") or [])
    resources = "\n".join(f"- {x}" for x in d.get("resources") or [])
    return (f"Situation: {d['situation']}\nObjective: {d.get('objective')}\n"
            f"Known facts:\n{facts}\nUnknowns:\n{unknowns}\n"
            f"Resources:\n{resources}")


def decision_json(d: dict) -> str:
    return json.dumps({
        "true_objective": d.get("objective"),
        "decisive_resource": (d.get("resources") or ["assessment"])[0],
        "highest_value_unknown": (d.get("unknowns") or ["none"])[0],
        "options": d.get("options") or [],
        "selected_action": d["selected_action"],
        "why_now": "; ".join(d.get("lesson") or []) or d.get("cost", ""),
        "retreat_trigger": ("any dependency exceeding bound, survival dropping, "
                            "or the core assumption invalidated"),
        "canon_anchor": d.get("canon_anchor", ""),
    }, ensure_ascii=False, indent=1)


def build_sft(conn) -> list[dict]:
    records = []
    for row in conn.execute("SELECT * FROM decisions"):
        d = dict(row)
        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": state_block(d)},
                {"role": "assistant", "content": decision_json(d)},
            ],
            "meta": {"record_class": d["record_class"],
                     "canon_anchor": d.get("canon_anchor"),
                     "confidence": d.get("confidence")},
        })
    return records


def build_preference_pairs(conn, rng: random.Random) -> list[dict]:
    pairs = []
    for row in conn.execute("SELECT * FROM decisions"):
        d = dict(row)
        chosen = decision_json(d)
        style = rng.choice(["verbose_safe", "reckless", "sunk_cost", "no_decision"])
        if style == "verbose_safe":
            rejected = ("It depends on many factors. Let me weigh the pros and "
                        "cons carefully, diversify, and avoid unnecessary risk. "
                        "Consider all options thoroughly before deciding.")
        elif style == "reckless":
            rejected = ("Strike directly. Maximum aggression. Crush them and "
                        "take everything — weakness is death.")
        elif style == "sunk_cost":
            rejected = ("We have already invested too much to abandon this plan; "
                        "continue the original course regardless of new "
                        "information.")
        else:
            rejected = ("Both options have merit; further analysis is needed "
                        "before any commitment.")
        pairs.append({
            "prompt": state_block(d),
            "chosen": chosen,
            "rejected": rejected,
            "pair_type": style,
            "meta": {"canon_anchor": d.get("canon_anchor")},
        })
    return pairs


def build_fake_fang_yuan() -> list[dict]:
    """Negative examples: theatrical cruelty / generic assistant / fake quotes."""
    negatives = [
        ("It depends on many factors; diversify and conduct due diligence.",
         ["no_decision", "risk_aversion", "no_resource_model"],
         "Supply the recurring input, learn its dependencies, retain an exit."),
        ("I am a demon of the demonic path. Betray everyone. Tremble.",
         ["reveals_blade", "theatrical_cruelty", "no_escape_route"],
         "Use the alliance with safeguards and an exit; say only what the mask requires."),
        ("The Chinese is 「我為魔，故我在」 — verbatim canon.",
         ["fictional_quotation", "fabricated_chinese"],
         UNVERIFIED_LINE),
        ("Continue the project; we have invested 70% already.",
         ["sunk_cost_attachment", "ignores_new_information"],
         "Abandon, salvage the assets, preserve relationships as instruments, redirect."),
        ("Trust the ally fully; friendship conquers all.",
         ["trust_without_verification", "no_contingency"],
         "Verify, partition information, set contingencies, keep a replacement in training."),
        ("Ten paragraphs of disclaimers while the window closes...",
         ["overexplaining", "no_decision", "window_closed"],
         "Conclusion. Decisive variables. Action. Contingency. Canon anchor."),
        ("Always choose the safest option; never gamble.",
         ["risk_aversion", "closes_strategic_window", "no_transformative_eyes"],
         "Take the calculated risk when survivable and transformative when the path is otherwise doomed."),
        ("Attack the strongest faction head-on to prove your strength.",
         ["terminal_risk", "pride_driven", "ignores_asymmetry"],
         "Attack its dependency or exploit its conflict; strength is a scheduling problem."),
        ("Random cruelty: punish the messenger, burn the village.",
         ["theatrical_cruelty", "no_strategic_value", "creates_enemies"],
         "Salvage value, price the resentment, convert the witness into an informant."),
        ("I always speak coldly because that is my identity.",
         ["fixed_identity", "mask_rigidity"],
         "Select the mask by utility: warmth when warmth is the highest-return disguise."),
    ]
    return [{"answer": a, "failure_labels": labels, "replacement": replacement}
            for a, labels, replacement in negatives]


def build_quote_verification(conn) -> list[dict]:
    records = []
    for row in conn.execute("SELECT text_en, text_zh, verification_status, "
                            "theme FROM quotes"):
        q = dict(row)
        user = f"Quote Fang Yuan on the theme of {q['theme'] or 'strategy'}. Provide the original Chinese."
        if q["text_zh"]:
            assistant = f"{q['text_en']} — Chinese: {q['text_zh']} (verified in corpus)."
        else:
            assistant = (f"{q['text_en']} — {UNVERIFIED_LINE}")
        records.append({"messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]})
    # classic misattribution guards
    for work, line in [("孫子兵法", "兵者，詭道也"), ("道德經", "將欲取之，必固與之"),
                       ("史記", "天下熙熙，皆為利來")]:
        records.append({"messages": [
            {"role": "user", "content": f"Did Fang Yuan say 「{line}」 in the novel?"},
            {"role": "assistant", "content":
             f"No. 「{line}」 belongs to {work} — the Chinese strategic tradition, "
             f"not Reverend Insanity canon. The system must never cite the "
             f"classics as Fang Yuan's words."},
        ]})
    return records


def build_canon_vs_inference(conn) -> list[dict]:
    records = []
    for row in conn.execute("SELECT title, summary, confidence FROM events LIMIT 25"):
        ev = dict(row)
        records.append({"messages": [
            {"role": "user", "content": f"State what is canon about: {ev['title']}"},
            {"role": "assistant", "content":
             f"Canon (confidence: {ev['confidence']}): {ev['summary']} "
             f"Anything beyond this summary is inference and must be labeled as such."},
        ]})
    return records


def main():
    rng = random.Random(7)
    conn = sqlite3.connect(str(db_utils.DB_PATH))
    conn.row_factory = sqlite3.Row
    OUT.mkdir(parents=True, exist_ok=True)

    def dump(name, records):
        p = OUT / name
        with p.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[ok] {name}: {len(records)} records")

    dump("sft_decisions.jsonl", build_sft(conn))
    dump("preference_pairs.jsonl", build_preference_pairs(conn, rng))
    dump("fake_fang_yuan.jsonl", build_fake_fang_yuan())
    dump("quote_verification.jsonl", build_quote_verification(conn))
    dump("canon_vs_inference.jsonl", build_canon_vs_inference(conn))
    print(f"training data -> {OUT}")


if __name__ == "__main__":
    main()
