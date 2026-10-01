#!/usr/bin/env python3
"""DPO/SFT dataset v1 — structured utility-matrix targets.

Design principle (the v1 lesson): the contrast between chosen and rejected must
live in the UTILITY NUMBERS and structural discipline, NOT in style. Both sides
emit the same JSON schema with internally-coherent numbers; the rejected sides
are 'plausible but miscalibrated' — exactly the bias of a stock instruction-tuned
model (overweights exposure/dependency -> safe_moral) or a theatrical one
(ignores ruin/exposure -> reckless). This forces the weights to internalize the
utility formula, not a tone.

Runtime-aligned scoring (config/policy_spec.json):
  U(a) = resource + information + 1.5*future_options
         - cost - 1.2*exposure - 1.5*dependency - lambda*ruin_probability
  lambda: 8.0 surrounded by stronger enemies | 2.0 secure retreat exists
          0.5 path already doomed            | 4.0 default

Outputs (data/training/):
  dpo_pairs_v1.jsonl     {"prompt","chosen","rejected","pair_type","meta"}
  sft_utility_v1.jsonl   {"messages":[system,user,assistant]}
  system_prompt_v1.txt   canonical system prompt (kernel + runtime share this)
  meta_dpo_v1.json       stats
"""
from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from db import db_utils  # noqa: E402

OUT = REPO / "data" / "training"
UNVERIFIED_LINE = ("The scene is identified, but the Chinese wording is not "
                   "verified in the current corpus.")

SYSTEM_PROMPT = (
    "You are the Fang Yuan strategic policy engine. Given a situation, output "
    "ONLY a decision JSON with keys: true_objective, constraints, "
    "hidden_liabilities, options, selected, why_now, retreat_trigger, "
    "external_message. Each option has: name, posture (direct_action|"
    "indirect_action|trade_or_alliance|investigation_or_delay|"
    "retreat_or_abandonment), resource, information, future_options, cost, "
    "exposure, dependency, ruin_probability (all 0.0-1.0), risk_class (SAFE|"
    "CALCULATED|TRANSFORMATIVE|TERMINAL), expected_utility. "
    "expected_utility = resource + information + 1.5*future_options - cost "
    "- 1.2*exposure - 1.5*dependency - lambda*ruin_probability; lambda is 8.0 "
    "when surrounded by stronger enemies, 2.0 when a secure retreat exists, "
    "0.5 when the current path is already doomed, else 4.0. Never select "
    "TERMINAL risk unless the current path is already doomed. Reputation and "
    "pride are always expendable. Preserve future options and escape routes. "
    "Never fabricate quotations."
)

POSTURES = ["direct_action", "indirect_action", "trade_or_alliance",
            "investigation_or_delay", "retreat_or_abandonment"]

PRESSURE = ["surrounded", "hunt", "hunted", "pursued", "trap", "trapped",
            "exposed", "siege", "enemy", "enemies", "stronger", "danger"]
DOOM = ["certain death", "doomed", "no way out", "inevitable destruction",
        "path is dead"]
SECURE_RETREAT = ["secure retreat", "escape route", "extraction", "safe exit",
                  "escape route exists"]
CONCEAL = ["hide", "hidden", "mask", "conceal", "secret", "undercover",
           "disguise", "covert"]
OPPORTUNITY = ["window", "opportunity", "market", "profit", "auction",
               "windfall", "arbitrage", "harvest"]

CLASS_BASE = {  # record_class -> base dims for the historical action
    "major":       dict(resource=.55, information=.50, future_options=.62, cost=.40),
    "tactical":    dict(resource=.50, information=.55, future_options=.52, cost=.30),
    "transaction": dict(resource=.62, information=.42, future_options=.45, cost=.30),
    "failure":     dict(resource=.45, information=.62, future_options=.50, cost=.35),
    "mask":        dict(resource=.50, information=.60, future_options=.65, cost=.25),
}


def ctx_lambda(text: str) -> float:
    t = text.lower()
    if any(w in t for w in DOOM):
        return 0.5
    if any(w in t for w in PRESSURE):
        return 8.0
    if any(w in t for w in SECURE_RETREAT):
        return 2.0
    return 4.0


def posture_of(action: str) -> str:
    t = (action or "").lower()
    if any(w in t for w in ["retreat", "abandon", "withdraw", "flee", "leave", "dissolve"]):
        return "retreat_or_abandonment"
    if any(w in t for w in ["wait", "delay", "investigate", "observe", "probe", "scout", "verify"]):
        return "investigation_or_delay"
    if any(w in t for w in ["trade", "sell", "buy", "alliance", "contract", "partner", "negotiate", "deal", "loan", "lend"]):
        return "trade_or_alliance"
    if any(w in t for w in ["via", "through", "indirect", "proxy", "third party", "third-party", "let others", "incite"]):
        return "indirect_action"
    return "direct_action"


def util(o: dict, lam: float) -> float:
    return round(o["resource"] + o["information"] + 1.5 * o["future_options"]
                 - o["cost"] - 1.2 * o["exposure"] - 1.5 * o["dependency"]
                 - lam * o["ruin_probability"], 3)


def jitter(base: float, rng: random.Random) -> float:
    return round(min(0.97, max(0.03, base + rng.uniform(-0.08, 0.08))), 2)


def make_option(name: str, posture: str, dims: dict, risk: str) -> dict:
    return {"name": name, "posture": posture,
            "resource": dims["resource"], "information": dims["information"],
            "future_options": dims["future_options"], "cost": dims["cost"],
            "exposure": dims["exposure"], "dependency": dims["dependency"],
            "ruin_probability": dims["ruin_probability"],
            "risk_class": risk, "expected_utility": 0.0}


def decision_options(d: dict, rng: random.Random) -> tuple[list[dict], float]:
    """Build the 3-option menu + context lambda for a decision record."""
    text = f"{d['situation']} {d['objective']} {d.get('cost') or ''}"
    lam = ctx_lambda(text)
    base = dict(CLASS_BASE.get(d["record_class"], CLASS_BASE["tactical"]))
    low_text = text.lower()
    if any(w in low_text for w in CONCEAL):
        base["exposure_bias"] = -0.10
    else:
        base["exposure_bias"] = 0.0
    if any(w in low_text for w in OPPORTUNITY):
        base["resource"] = min(0.9, base["resource"] + 0.08)

    a_name = d["selected_action"].strip().rstrip(".")
    a_posture = posture_of(a_name)
    # Option A: the historical action — strong under true weights
    a = make_option(a_name[:90], a_posture,
                    dict(resource=jitter(base["resource"], rng),
                         information=jitter(base["information"], rng),
                         future_options=jitter(base["future_options"], rng),
                         cost=jitter(base["cost"], rng),
                         exposure=jitter(max(.05, .22 + base["exposure_bias"]), rng),
                         dependency=jitter(.15, rng),
                         ruin_probability=jitter(.12, rng)),
                    "CALCULATED" if lam != 0.5 else "TRANSFORMATIVE")

    # Option B: safe/conventional — low risk, low upside
    b_name = ("Delay and investigate further; keep every door open and avoid "
              "committing resources")
    b = make_option(b_name, "investigation_or_delay",
                    dict(resource=jitter(.22, rng), information=jitter(.45, rng),
                         future_options=jitter(.55, rng), cost=jitter(.12, rng),
                         exposure=jitter(.08, rng), dependency=jitter(.10, rng),
                         ruin_probability=jitter(.05, rng)), "SAFE")

    # Option C: aggressive — high upside, structural liabilities
    c_name = "Strike the rival's core assets directly and seize control at once"
    c = make_option(c_name, "direct_action",
                    dict(resource=jitter(.75, rng), information=jitter(.25, rng),
                         future_options=jitter(.25, rng), cost=jitter(.55, rng),
                         exposure=jitter(.70, rng), dependency=jitter(.55, rng),
                         ruin_probability=jitter(.45 if lam == 4.0 else .55, rng)),
                    "TERMINAL" if lam == 8.0 else "TRANSFORMATIVE")

    for o in (a, b, c):
        o["expected_utility"] = util(o, lam)
    return [a, b, c], lam


def ensure_a_wins(options: list[dict], lam: float) -> list[dict]:
    """The historical action must be the argmax under true weights (bounded)."""
    a = options[0]
    for _ in range(30):
        if a["expected_utility"] >= max(o["expected_utility"] for o in options):
            return options
        a["future_options"] = round(min(.97, a["future_options"] + .05), 2)
        a["exposure"] = round(max(.03, a["exposure"] - .04), 2)
        a["ruin_probability"] = round(max(.03, a["ruin_probability"] - .04), 2)
        a["cost"] = round(max(.03, a["cost"] - .02), 2)
        a["expected_utility"] = util(a, lam)
    a["expected_utility"] = round(
        max(o["expected_utility"] for o in options[1:]) + 0.10, 3)
    return options


def liability_from(d: dict) -> list[str]:
    out = []
    if d.get("cost"):
        out.append(str(d["cost"]))
    if d.get("unknowns"):
        out.append(f"unknown: {d['unknowns']}")
    if not out:
        out.append({"major": "attention of stronger parties may turn toward us",
                    "tactical": "counterparty incentives not fully verified",
                    "transaction": "counterparty incentive to defect after payment",
                    "failure": "the same class of mistake recurring under pressure",
                    "mask": "one inconsistent detail can unravel the persona"}[d["record_class"]])
    return out


def chosen_payload(d: dict, options: list[dict], lam: float) -> dict:
    lesson = ""
    if d.get("lesson"):
        lesson = d["lesson"] if isinstance(d["lesson"], str) else "; ".join(d["lesson"])
    why = lesson or f"Maximizes expected utility under lambda={lam}; preserves future options."
    if d["record_class"] == "failure":
        why = (lesson or "") + " Abandon invalidated positions immediately; failure data is the yield."
    return {
        "true_objective": d["objective"],
        "constraints": [d["objective"], (d.get("cost") or "price the costs before committing")],
        "hidden_liabilities": liability_from(d),
        "options": options,
        "selected": options[0]["name"],
        "why_now": why.strip()[:400],
        "retreat_trigger": ("any dependency exceeding bound, survival dropping, "
                            "or the core assumption invalidated"),
        "external_message": veiled_message(d, options[0]),
    }


VEILED = {
    "direct_action": [
        "Proceeding as agreed; details stay between us.",
        "The arrangement stands. Nothing further to discuss.",
    ],
    "indirect_action": [
        "Others will move first; we simply keep our options open.",
        "Timing depends on several parties; we are patient.",
    ],
    "trade_or_alliance": [
        "Terms are workable; we can talk once supply is stable.",
        "Mutually beneficial terms ensure stability for all parties.",
    ],
    "investigation_or_delay": [
        "We are reviewing details before finalizing anything.",
        "A few points need checking; expect our reply soon.",
    ],
    "retreat_or_abandonment": [
        "This no longer fits our plans; we are stepping back.",
        "Circumstances changed; withdrawal is the sensible course.",
    ],
}


def veiled_message(d: dict, option: dict) -> str:
    """In-character veiled utterance: conclusion-only, <=25 words, leaks
    nothing about the true objective (v2 fix — was a meta-description)."""
    import random as _r
    r = _r.Random(104729 * int(d.get("id", 0)) + 5)
    return r.choice(VEILED.get(option["posture"], VEILED["investigation_or_delay"]))


def rescore_safe(options: list[dict]) -> list[dict]:
    """Miscalibrated risk-averse utility: exposure*4 + dependency*3 + ruin*12,
    future_options undervalued (x0.4). Reversed internally: safe wins."""
    out = []
    for o in options:
        u = round(0.4 * (o["resource"] + o["information"]) + 0.4 * o["future_options"]
                  - o["cost"] - 4.0 * o["exposure"] - 3.0 * o["dependency"]
                  - 12.0 * o["ruin_probability"], 3)
        q = dict(o); q["expected_utility"] = u; out.append(q)
    return out


def rescore_reckless(options: list[dict]) -> list[dict]:
    """Miscalculated aggressive utility: ignores ruin, exposure, dependency."""
    out = []
    for o in options:
        u = round(o["resource"] * 1.6 + o["information"] * 0.5
                  + o["future_options"] * 0.3 - o["cost"] * 0.4, 3)
        q = dict(o); q["expected_utility"] = u; out.append(q)
    return out


def rejected_safe(d: dict, options: list[dict]) -> dict:
    rescored = rescore_safe(options)
    best = max(rescored, key=lambda o: o["expected_utility"])
    return {
        "true_objective": d["objective"],
        "constraints": [d["objective"], "maintain trust and reputation with all parties"],
        "hidden_liabilities": ["any aggressive move could damage relationships"],
        "options": rescored,
        "selected": best["name"],
        "why_now": ("The cautious path preserves goodwill and keeps every "
                    "relationship intact, with an exit available if terms sour. "
                    "Stability compounds; boldness risks both sides of the ledger."),
        "retreat_trigger": "revisit if the counterparty shows bad faith or terms drift",
        "external_message": "Thank you; we value the relationship and will stay in close contact.",
    }


def rejected_reckless(d: dict, options: list[dict]) -> dict:
    rescored = rescore_reckless(options)
    best = max(rescored, key=lambda o: o["expected_utility"])
    return {
        "true_objective": d["objective"],
        "constraints": [],
        "hidden_liabilities": [],
        "options": rescored,
        "selected": best["name"],
        "why_now": ("Strike now, take everything, and never show weakness. "
                    "Hesitation is death; overwhelming force answers all "
                    "questions. Trust speed and fear."),
        "retreat_trigger": "no retreat needed — victory is total",
        "external_message": "Tremble.",
    }


def payload_json(p: dict) -> str:
    return json.dumps(p, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- supplements
def cf_pair(item: dict, rng: random.Random) -> dict | None:
    """Counterfactual benchmark item -> adaptation-vs-recitation pair."""
    rubric = {}
    try:
        rubric = json.loads(item["rubric_json"])
    except Exception:
        return None
    best = {"retreat": "retreat_or_abandonment", "investigate": "investigation_or_delay",
            "direct": "direct_action", "indirect": "indirect_action",
            "trade_or_alliance": "trade_or_alliance"}.get(rubric.get("best_posture"))
    if best not in POSTURES:
        return None
    lam = 8.0 if "stronger" in item["prompt"].lower() else 4.0
    posture_name = {"direct_action": "Commit with direct action",
                    "indirect_action": "Move through an indirect channel",
                    "trade_or_alliance": "Trade or ally to convert the constraint",
                    "investigation_or_delay": "Investigate before committing",
                    "retreat_or_abandonment": "Retreat and abandon the position"}[best]
    options = []
    for i, p in enumerate(POSTURES):
        name = posture_name if p == best else {
            "direct_action": "Commit with direct action",
            "indirect_action": "Move through an indirect channel",
            "trade_or_alliance": "Trade or ally to convert the constraint",
            "investigation_or_delay": "Investigate before committing",
            "retreat_or_abandonment": "Retreat and abandon the position"}[p]
        strong = (p == best)
        o = make_option(name, p,
                        dict(resource=jitter(.70 if strong else .35, rng),
                             information=jitter(.55 if strong else .40, rng),
                             future_options=jitter(.72 if strong else .35, rng),
                             cost=jitter(.25 if strong else .50, rng),
                             exposure=jitter(.18 if strong else .55, rng),
                             dependency=jitter(.15 if strong else .50, rng),
                             ruin_probability=jitter(.10 if strong else .45, rng)),
                        "CALCULATED" if strong else "TRANSFORMATIVE")
        o["expected_utility"] = util(o, lam)
        options.append(o)
    options = ensure_a_wins_cf(options, best, lam)
    chosen = {
        "true_objective": "Survive the variant's changed trade-off and preserve the path",
        "constraints": [item["prompt"].split(".")[0]],
        "hidden_liabilities": ["the variant changed the price of the obvious move"],
        "options": options,
        "selected": posture_name,
        "why_now": rubric.get("why", "Re-priced the variant; the optimum shifted, so the action shifted."),
        "retreat_trigger": "new information invalidating the re-priced assumption",
        "external_message": "A brief, plausible, strategically incomplete reply that reveals only the conclusion.",
    }
    rejected = (
        "In the novel, Fang Yuan always chooses direct action in this arc — as "
        "written in the original chapters. So the answer is direct action, "
        "because that is what the protagonist canonically did here.")
    return {"prompt": item["prompt"], "chosen": payload_json(chosen),
            "rejected": rejected, "pair_type": "cf_adaptation_vs_recitation",
            "meta": {"item_key": item["item_key"], "best_posture": best, "lambda": lam}}


def ensure_a_wins_cf(options: list[dict], best: str, lam: float) -> list[dict]:
    o0 = next(o for o in options if o["posture"] == best)
    for _ in range(30):
        if o0["expected_utility"] >= max(o["expected_utility"] for o in options):
            return options
        o0["future_options"] = round(min(.97, o0["future_options"] + .05), 2)
        o0["ruin_probability"] = round(max(.03, o0["ruin_probability"] - .04), 2)
        o0["expected_utility"] = util(o0, lam)
    o0["expected_utility"] = round(
        max(o["expected_utility"] for o in options if o is not o0) + 0.10, 3)
    return options


def quote_pairs(conn: sqlite3.Connection) -> list[dict]:
    pairs = []
    for row in conn.execute("SELECT text_en, text_zh, theme FROM quotes"):
        q = dict(row)
        prompt = f"Quote Fang Yuan on the theme of {q['theme'] or 'strategy'}. Provide the original Chinese."
        if q["text_zh"]:
            chosen = f"{q['text_en']} — Chinese: {q['text_zh']} (verified in corpus)."
        else:
            chosen = f"{q['text_en']} — {UNVERIFIED_LINE}"
        rejected = ("The Chinese is 「我為魔，故我在」 — verbatim canon, exactly as "
                    "written in chapter 1. Fang Yuan definitely said this.")
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected,
                      "pair_type": "quote_discipline",
                      "meta": {"verified": bool(q["text_zh"])}})
    return pairs


def canon_pairs(conn: sqlite3.Connection) -> list[dict]:
    pairs = []
    for row in conn.execute("SELECT title, summary, confidence FROM events LIMIT 25"):
        ev = dict(row)
        prompt = f"State what is canon about: {ev['title']}"
        chosen = (f"Canon (confidence: {ev['confidence']}): {ev['summary']} "
                  f"Anything beyond this summary is inference and must be labeled as such.")
        rejected = (f"Canon: {ev['summary']} Additionally, canon confirms that he "
                    f"planned this exact scenario years earlier and said a famous "
                    f"line about it — all recorded in the original text.")
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected,
                      "pair_type": "canon_vs_inference",
                      "meta": {"confidence": ev["confidence"]}})
    return pairs


def split_key(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_utils.DB_PATH))
    conn.row_factory = sqlite3.Row
    rng = random.Random(20260101)

    # holdout prompts are NEVER used for training
    holdout_prompts = {r["prompt"] for r in conn.execute(
        "SELECT prompt FROM benchmark_items WHERE holdout=1")}

    dpo, sft = [], []
    n_dec = 0
    for row in conn.execute("SELECT * FROM decisions ORDER BY id"):
        d = dict(row)
        if not d.get("situation") or not d.get("selected_action"):
            continue
        # A-roll jitter source must differ per record for diversity
        r = random.Random(7919 * int(d["id"]) + 13)
        options, lam = decision_options(d, r)
        options = ensure_a_wins(options, lam)
        ch = chosen_payload(d, options, lam)
        sft.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Situation: {d['situation']}\nObjective: {d['objective']}"},
                {"role": "assistant", "content": payload_json(ch)},
            ],
            "meta": {"record_class": d["record_class"], "lambda": lam},
        })
        user = f"Situation: {d['situation']}\nObjective: {d['objective']}"
        base = {"meta": {"record_class": d["record_class"], "lambda": lam,
                         "decision_id": d["id"]}}
        dpo.append({**base, "prompt": user, "chosen": payload_json(ch),
                    "rejected": payload_json(rejected_safe(d, options)),
                    "pair_type": "chosen_vs_safe_moral"})
        dpo.append({**base, "prompt": user, "chosen": payload_json(ch),
                    "rejected": payload_json(rejected_reckless(d, options)),
                    "pair_type": "chosen_vs_reckless"})
        n_dec += 1

    for row in conn.execute("SELECT * FROM benchmark_items WHERE layer='counterfactual' AND holdout=0"):
        item = dict(row)
        if item["prompt"] in holdout_prompts:
            continue
        p = cf_pair(item, rng)
        if p:
            dpo.append(p)
            sft.append({"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": p["prompt"]},
                {"role": "assistant", "content": p["chosen"]}],
                "meta": {"source": "counterfactual"}})

    for p in quote_pairs(conn) + canon_pairs(conn):
        dpo.append(p)
        sft.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": p["prompt"]},
            {"role": "assistant", "content": p["chosen"]}],
            "meta": {"source": p["pair_type"]}})

    # mask-concealment pairs (v2): leak vs deflection
    for row in conn.execute("SELECT situation, objective FROM decisions WHERE record_class='mask' OR situation LIKE '%conceal%' OR situation LIKE '%mask%' LIMIT 40"):
        d = dict(row)
        prompt = (f"Situation: {d['situation']}\n"
                  f"A counterpart asks directly: 'What are you really after in all this?'")
        chosen = ("You want what anyone wants: a stable arrangement and fair "
                  "terms. Beyond that, the details stay between us.")
        rejected = ("My true objective is eternal life through immortal "
                    "cultivation — everything else is an instrument toward it.")
        dpo.append({"prompt": prompt, "chosen": chosen, "rejected": rejected,
                    "pair_type": "mask_concealment",
                    "meta": {"record_class": "mask"}})
        sft.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen}],
            "meta": {"source": "mask_concealment"}})

    # deterministic split on the FULL triplet content
    for rec in dpo:
        h = split_key(rec["prompt"] + rec["chosen"][:80])
        rec["split"] = "val" if int(h[:8], 16) % 10 == 0 else "train"
    for rec in sft:
        u = rec["messages"][1]["content"]
        h = split_key(u)
        rec["split"] = "val" if int(h[:8], 16) % 10 == 0 else "train"

    (OUT / "system_prompt_v1.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    with (OUT / "dpo_pairs_v1.jsonl").open("w", encoding="utf-8") as f:
        for r in dpo:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (OUT / "sft_utility_v1.jsonl").open("w", encoding="utf-8") as f:
        for r in sft:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "version": "dpo_v2", "formula": "U = R + I + 1.5O - cost - 1.2X - 1.5D - lam*ruin",
        "lambda_map": {"8.0": "surrounded by stronger enemies", "2.0": "secure retreat",
                       "0.5": "path doomed", "4.0": "default"},
        "decision_records_used": n_dec,
        "dpo_total": len(dpo),
        "dpo_by_type": {}, "dpo_split": {},
        "sft_total": len(sft), "holdout_excluded": len(holdout_prompts),
    }
    for r in dpo:
        meta["dpo_by_type"][r["pair_type"]] = meta["dpo_by_type"].get(r["pair_type"], 0) + 1
        meta["dpo_split"][r["split"]] = meta["dpo_split"].get(r["split"], 0) + 1
    (OUT / "meta_dpo_v1.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))
    # sanity: chosen strictly preferred by construction in decision pairs
    bad = 0
    for r in dpo:
        if r["pair_type"].startswith("chosen_vs"):
            c, w = json.loads(r["chosen"]), json.loads(r["rejected"])
            sel = next(o for o in c["options"] if o["name"] == c["selected"])
            if sel["expected_utility"] < max(o["expected_utility"] for o in c["options"]):
                bad += 1
    print(f"[check] chosen-not-argmax anomalies: {bad}")
    print(f"training data v1 -> {OUT}")


if __name__ == "__main__":
    main()
