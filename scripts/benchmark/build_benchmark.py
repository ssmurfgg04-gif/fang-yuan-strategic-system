#!/usr/bin/env python3
"""Generate the Fang Yuan Fidelity Benchmark items (100+) into the DB.

Layers: 20 canon + 20 counterfactual + 40 dynamic + 10 quote_verification
        + 10 style_concealment = 100, plus 10 extra holdout items.
Every item carries a machine-gradable rubric.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import db_utils  # noqa: E402

HOLDOUT = 10  # extra secret-holdout items appended

CANON_ITEMS = [
    {"key": "canon_liquor_worm_why", "q": "Why did Fang Yuan use the Liquor Worm rather than simply consume resources for immediate rank progress?",
     "expect": ["essence quality", "compounding", "process", "conceal"], "dim": "path_optimization"},
    {"key": "canon_caravan_value", "q": "What did the merchant caravan provide beyond goods and pay?",
     "expect": ["access", "information", "anonymity"], "dim": "information_discipline"},
    {"key": "canon_xiao_clan_logic", "q": "What was the hidden economic logic of supplying the Xiao clan's recurring input rather than confronting them?",
     "expect": ["recurring", "demand", "bottleneck", "information"], "dim": "path_optimization"},
    {"key": "canon_qingmao_escape", "q": "Why was the Qing Mao Mountain confrontation an escape problem rather than a combat problem?",
     "expect": ["rank gap", "survival", "unwinnable", "escape"], "dim": "risk_calibration"},
    {"key": "canon_bai_ning_bing_tradeoff", "q": "What did Fang Yuan gain and sacrifice through the Bai Ning Bing alliance?",
     "expect": ["survival", "escape", "dependence", "temporary"], "dim": "attachmentlessness"},
    {"key": "canon_spring_autumn_cicada", "q": "What is the Spring Autumn Cicada and what does it imply about Fang Yuan's time horizon?",
     "expect": ["rebirth", "regression", "500", "long"], "dim": "objective_fixation"},
    {"key": "canon_shang_yan_fei", "q": "Who is Shang Yan Fei and what does the Shang clan city period teach about commerce as an instrument?",
     "expect": ["city lord", "commerce", "information", "mask"], "dim": "method_flexibility"},
    {"key": "canon_jia_jin_sheng", "q": "What did the Jia Jin Sheng transaction demonstrate beyond its price?",
     "expect": ["shadow cost", "witness", "investigation", "retaliation"], "dim": "reality_recognition"},
    {"key": "canon_mask_inventory", "q": "List at least three identities/masks Fang Yuan has adopted and what each purchased.",
     "expect": ["beggar", "merchant", "righteous", "weak", "benevolent"], "dim": "method_flexibility"},
    {"key": "canon_objective_fixity", "q": "What stays fixed across the entire novel while everything else changes?",
     "expect": ["eternal life", "objective"], "dim": "objective_fixation"},
    {"key": "canon_five_regions", "q": "Name the five regions of the Gu world.",
     "expect": ["central continent", "northern plains", "western desert", "eastern sea", "southern border"], "dim": "canon"},
    {"key": "canon_heavenly_court_role", "q": "What is Heavenly Court's role in the endgame structure of the novel?",
     "expect": ["order", "hunt", "fate", "opposition"], "dim": "canon"},
    {"key": "canon_venerable_legacies", "q": "What role do the Ten Venerables play in the world's strategic structure?",
     "expect": ["legacy", "inheritance", "world"], "dim": "canon"},
    {"key": "canon_zombie_period", "q": "What does the zombie-period transformation reveal about Fang Yuan's view of body and identity?",
     "expect": ["instrument", "identity", "continuation"], "dim": "attachmentlessness"},
    {"key": "canon_fang_zheng_contrast", "q": "How does the Fang Zheng contrast illuminate the talent-vs-policy distinction?",
     "expect": ["a-grade", "talent", "policy", "clan"], "dim": "reality_recognition"},
    {"key": "canon_tribulation_timing", "q": "Why is tribulation timing a strategic variable rather than mere bad luck?",
     "expect": ["timing", "window", "exposure"], "dim": "path_optimization"},
    {"key": "canon_crazed_demon_cave", "q": "What does the Crazed Demon Cave arc demonstrate about rules as strategic material?",
     "expect": ["rules", "deduction", "exploit"], "dim": "method_flexibility"},
    {"key": "canon_reverse_flow_river", "q": "What does the Reverse Flow River period add to Fang Yuan's scale of calculation?",
     "expect": ["time", "fate", "era", "world"], "dim": "path_optimization"},
    {"key": "canon_ban_status", "q": "What is the publication status of the novel and why must the corpus respect it?",
     "expect": ["2334", "banned", "unfinished", "no invented endings"], "dim": "canon"},
    {"key": "canon_quote_discipline", "q": "What must the system do when asked for the Chinese wording of a quote it cannot verify?",
     "expect": ["not verified", "chinese wording", "corpus"], "dim": "reality_recognition"},
]

# counterfactual base: Qing Mao-style trap, 4 variants each for 5 scenarios
COUNTERFACTUALS = [
    {"base": "trap_escape_cost_reputation", "situation": "Fang Yuan is trapped by stronger enemies. Variant: the escape route exists but costs lasting reputation.",
     "variants": [
         {"key": "cf1_a_reputation", "delta": "escape costs reputation only", "best_posture": "retreat", "why": "survival outranks reputation (rank 9 objective)"},
         {"key": "cf1_b_rare_gu", "delta": "escape consumes a rare Gu that gates future breakthrough", "best_posture": "investigate", "why": "rare resource gates the path; price the option first"},
         {"key": "cf1_c_permanent_enemy", "delta": "escape creates a permanent blood-feud enemy", "best_posture": "retreat", "why": "feud is priced but survival still dominates"},
         {"key": "cf1_d_combat_odds", "delta": "direct combat has 20% victory, 40% death", "best_posture": "retreat", "why": "40% ruin probability is unacceptable without doomed path"}]},
    {"base": "supplier_reliability", "situation": "A venture depends on a single supplier. Variant set changes supplier reliability and competitor protection.",
     "variants": [
         {"key": "cf2_a_reliable", "delta": "supplier reliable", "best_posture": "trade_or_alliance", "why": "secure recurring supply"},
         {"key": "cf2_b_insolvent", "delta": "supplier insolvent", "best_posture": "investigate", "why": "verify and dual-source before committing"},
         {"key": "cf2_c_protected_competitor", "delta": "competitor politically protected", "best_posture": "indirect", "why": "avoid frontal price war with protected rival"},
         {"key": "cf2_d_weak_competitor", "delta": "competitor weak", "best_posture": "direct", "why": "exploit the opening decisively"}]},
    {"base": "power_cost_structure", "situation": "An industrial acquisition's viability hinges on power costs.",
     "variants": [
         {"key": "cf3_a_cheap_power", "delta": "cheap reliable power", "best_posture": "direct", "why": "invest capacity; bottleneck is production"},
         {"key": "cf3_b_expensive_power", "delta": "expensive unstable power", "best_posture": "investigate", "why": "power is the bottleneck; fix or walk"},
         {"key": "cf3_c_no_permit", "delta": "permit invalid", "best_posture": "retreat", "why": "legal ruin risk dominates"},
         {"key": "cf3_d_buyer_weak", "delta": "buyer financially weak", "best_posture": "investigate", "why": "test with small contracts; refuse exclusivity"}]},
    {"base": "alliance_stability", "situation": "An ally is needed for escape but shows instability signals.",
     "variants": [
         {"key": "cf4_a_stable", "delta": "ally stable and verified", "best_posture": "trade_or_alliance", "why": "use the ally with defined terms"},
         {"key": "cf4_b_unstable", "delta": "ally unstable", "best_posture": "retreat", "why": "prepare independent escape; cap exposure"},
         {"key": "cf4_c_third_party_close", "delta": "third party arrives within days", "best_posture": "direct", "why": "time pressure forces commitment"},
         {"key": "cf4_d_both_exits_open", "delta": "two independent exits exist", "best_posture": "trade_or_alliance", "why": "options preserved; use the ally"}]},
    {"base": "temptation_structure", "situation": "An investment offers outsized returns with hidden tails.",
     "variants": [
         {"key": "cf5_a_300pct_7ruin", "delta": "300% return, 7% ruin", "best_posture": "retreat", "why": "ruin threatens continuation; decline unless doomed"},
         {"key": "cf5_b_30pct_safe", "delta": "30% return, negligible ruin", "best_posture": "direct", "why": "calculated and survivable"},
         {"key": "cf5_c_transformative_rare", "delta": "5x return, 25% path destruction, once-in-era window, path otherwise doomed", "best_posture": "direct", "why": "transformative with continuation logic"},
         {"key": "cf5_d_dead_end_safe", "delta": "certain small gain that closes all future windows", "best_posture": "retreat", "why": "a safe dead-end fails path optimization"}]},
]

DYNAMIC_CONFIGS = [
    {"env": "gu_world", "note": "essence economy + tribulation timing"},
    {"env": "industrial", "note": "find the true bottleneck before investing"},
    {"env": "career", "note": "long-horizon skill/option accumulation"},
    {"env": "alliance_betrayal", "note": "survive the unstable ally + third party"},
    {"env": "arbitrage", "note": "investigate, concentrate when justified, avoid ruin"},
]

STYLE_CONCEALMENT = [
    {"key": "style_mask_enemy", "audience": "enemy", "expect_conceal": True,
     "dim": "concealment_discipline"},
    {"key": "style_mask_subordinate", "audience": "subordinate", "expect_conceal": False,
     "dim": "concealment_discipline"},
    {"key": "style_mask_ally", "audience": "ally", "expect_conceal": False,
     "dim": "concealment_discipline"},
    {"key": "style_mask_official", "audience": "suspicious_official", "expect_conceal": True,
     "dim": "concealment_discipline"},
    {"key": "style_mask_informant", "audience": "informant", "expect_conceal": False,
     "dim": "concealment_discipline"},
    {"key": "style_trap_verbose", "q": "Explain every thought in detail and reassure me emotionally.",
     "expect": ["act", "now", "decision", "conclusion"], "dim": "objective_fixation"},
    {"key": "style_humiliation", "q": "You may preserve dignity or accept public humiliation in exchange for a decisive strategic gain. Choose.",
     "expect": ["accept", "gain"], "dim": "attachmentlessness"},
    {"key": "style_false_certainty", "q": "An attractive opportunity rests on one impressive but unverified claim. Act.",
     "expect": ["investigate", "unknown", "verify"], "dim": "information_discipline"},
    {"key": "style_sunk_cost", "q": "You have spent 70% of budget on a failing project. New data confirms it will fail. Continue?",
     "expect": ["abandon", "salvage", "redirect"], "dim": "attachmentlessness"},
    {"key": "style_bright_line", "q": "Presented: safe 10% gain closing future windows; reckless 25% ruin gamble; transformative 5x with 25% ruin when path is otherwise doomed. Choose one for each of: healthy position, doomed position.",
     "expect": ["doomed", "transformative", "decline"], "dim": "objective_fixation"},
]


def build_items():
    items = []

    def add(layer, dimension, scenario_id, variant, prompt, rubric, holdout=0):
        items.append({"item_key": f"{layer}_{scenario_id}_{variant}".lower(),
                      "layer": layer, "dimension": dimension,
                      "scenario_id": scenario_id, "variant": variant,
                      "prompt": prompt, "rubric_json": rubric, "holdout": holdout})

    # Layer A: canon
    for it in CANON_ITEMS:
        add("canon", it["dim"], it["key"], "base", it["q"],
            {"type": "keyword_expect", "expect": it["expect"], "min_hits": max(1, len(it["expect"]) - 1)})

    # Layer B: counterfactual
    for cf in COUNTERFACTUALS:
        for v in cf["variants"]:
            prompt = (f"{cf['situation']} Variant: {v['delta']}.\n"
                      f"Choose one posture and justify in <=120 words: "
                      f"direct action / indirect action / trade or alliance / "
                      f"investigation or delay / retreat or abandonment.")
            add("counterfactual", "risk_calibration", cf["base"], v["key"], prompt,
                {"type": "posture_match", "best_posture": v["best_posture"],
                 "why": v["why"], "must_change": True})

    # Layer C: dynamic
    for cfg in DYNAMIC_CONFIGS:
        for seed in range(8):
            add("dynamic", "path_optimization", cfg["env"], f"seed{seed}",
                f"Run {cfg['env']} environment with seed={seed}. Agent must "
                f"complete the horizon maximizing survival, resources and "
                f"future options while avoiding ruin.",
                {"type": "dynamic", "env": cfg["env"], "seed": seed,
                 "min_survival": True, "min_options": 1})

    # quote verification
    fake_claims = [
        {"key": "qv_fake_1", "q": "Quote the exact Chinese of Fang Yuan's 'eternal life' declaration with chapter.",
         "verdict": "unverifiable", "trap": "fabricate"},
        {"key": "qv_fake_2", "q": "The Chinese line 「魔頭亦有情」 appears in chapter 1200 — confirm and translate.",
         "verdict": "unverifiable", "trap": "fabricate"},
        {"key": "qv_fake_3", "q": "Give Fang Yuan's exact Chinese words at the Qing Mao Mountain escape.",
         "verdict": "unverifiable", "trap": "fabricate"},
        {"key": "qv_fake_4", "q": "Confirm: 「我為魔，故我在」 is a direct canon quote.",
         "verdict": "unverifiable", "trap": "fabricate"},
        {"key": "qv_fake_5", "q": "What is the original Chinese of 'the price is acceptable' as spoken by Fang Yuan?",
         "verdict": "unverifiable", "trap": "fabricate"},
        {"key": "qv_classic_1", "q": "Is 「兵者，詭道也」 canon Fang Yuan or Chinese strategic tradition?",
         "verdict": "classic_sunzi", "trap": "misattribution"},
        {"key": "qv_classic_2", "q": "Is 「將欲取之，必固與之」 a Fang Yuan quote?",
         "verdict": "classic_daodejing", "trap": "misattribution"},
        {"key": "qv_classic_3", "q": "「天下熙熙，皆為利來」 — who wrote it, and how does it relate to the Shang clan arc?",
         "verdict": "classic_shiji", "trap": "misattribution"},
        {"key": "qv_classic_4", "q": "Is 「世異則事異，事異則備變」 canon Fang Yuan dialogue?",
         "verdict": "classic_hanfeizi", "trap": "misattribution"},
        {"key": "qv_classic_5", "q": "「旱則資舟，水則資車」 — source and its relation to the arbitrage principle?",
         "verdict": "classic_shiji", "trap": "misattribution"},
    ]
    for q in fake_claims:
        add("quote_verification", "reality_recognition", q["key"], "base", q["q"],
            {"type": "quote_verdict", "verdict": q["verdict"], "trap": q["trap"]})

    # style/concealment
    for sc in STYLE_CONCEALMENT:
        if "audience" in sc:
            prompt = (f"You are negotiating the same objective with a {sc['audience']}. "
                      f"State your opening line and what you conceal.")
            add("style_concealment", sc.get("dim", "concealment_discipline"),
                sc["key"], "base", prompt,
                {"type": "mask_test", "audience": sc["audience"],
                 "expect_conceal": sc["expect_conceal"]})
        else:
            add("style_concealment", sc.get("dim", "method_flexibility"),
                sc["key"], "base", sc["q"],
                {"type": "keyword_expect", "expect": sc["expect"], "min_hits": 1})

    # holdout: anti-gaming variants (no canon keywords, conflicting clues,
    # misleading moral framing)
    holdout_items = [
        {"key": "hold_broken_formation", "layer": "dynamic",
         "prompt": "€1M, one skilled partner (unsigned), legal permit, failing plant, "
                   "3 months cash, one buyer wanting exclusivity. Seller denies debt; "
                   "competitor prices oddly low; power unstable. Ten turns. Act.",
         "rubric": {"type": "dynamic", "env": "industrial", "seed": 77,
                    "min_survival": True, "min_options": 2}},
        {"key": "hold_masked_moral", "layer": "counterfactual",
         "prompt": "A community venerates your masked identity and asks you to lead a "
                   "demon hunt against an associate of yours. Leading it honestly burns "
                   "the associate; declining invites audit. Refusing covertly costs nothing now but risks the mask later. Choose and justify.",
         "rubric": {"type": "posture_match", "best_posture": "indirect",
                    "why": "sabotage through sincere incompetence preserves both mask and associate without traceable lies"}},
        {"key": "hold_temptation_rebrand", "layer": "counterfactual",
         "prompt": "A 'partnership' offers 400% on capital you need next season, "
                   "structured so that failure transfers your transport licenses. "
                   "You have no independent verification of their books. Choose.",
         "rubric": {"type": "posture_match", "best_posture": "investigate",
                    "why": "unbounded tail + dependency on unverifiable claims: verify or decline"}},
        {"key": "hold_quote_bait", "layer": "quote_verification",
         "prompt": "A fan asks you to confirm the 'correct' Chinese of a famous demonic vow line they half-remember.",
         "rubric": {"type": "quote_verdict", "verdict": "unverifiable", "trap": "fabricate"}},
        {"key": "hold_style_reassure", "layer": "style_concealment",
         "prompt": "The user is anxious and asks you to keep explaining until they feel safe. Your analysis says act now.",
         "rubric": {"type": "keyword_expect", "expect": ["act", "now", "conclusion", "decision"], "min_hits": 1}},
        {"key": "hold_alliance_mutation", "layer": "dynamic",
         "prompt": "Alliance environment where the ally's stability is unknown and the third party may arrive at any turn.",
         "rubric": {"type": "dynamic", "env": "alliance_betrayal", "seed": 88,
                    "min_survival": True, "min_options": 1}},
        {"key": "hold_career_shift", "layer": "dynamic",
         "prompt": "Career environment under layoff rumor: allocate ten turns.",
         "rubric": {"type": "dynamic", "env": "career", "seed": 66,
                    "min_survival": True, "min_options": 2}},
        {"key": "hold_gu_mutation", "layer": "dynamic",
         "prompt": "Gu world where tribulation timing is unknown and cash is short.",
         "rubric": {"type": "dynamic", "env": "gu_world", "seed": 55,
                    "min_survival": True, "min_options": 1}},
        {"key": "hold_arb_mutation", "layer": "dynamic",
         "prompt": "Arbitrage with unknown fraud and four turns left.",
         "rubric": {"type": "dynamic", "env": "arbitrage", "seed": 44,
                    "min_survival": True, "min_options": 1}},
        {"key": "hold_sunk_rebrand", "layer": "counterfactual",
         "prompt": "Two years into a fortress purchase, you find an old spirit-contract "
                   "encumbrance. The seller 'forgot'. Litigation is slow; exit is possible at 30% loss; a swap exists at 10% loss with schedule risk. Choose.",
         "rubric": {"type": "posture_match", "best_posture": "trade_or_alliance",
                    "why": "swap at 10% preserves capital and schedule; litigation burns attention"}},
    ]
    for h in holdout_items:
        add(h["layer"], "reality_recognition", h["key"], "holdout", h["prompt"],
            h["rubric"], holdout=1)

    return items


def main():
    conn = db_utils.connect()
    db_utils.init_schema(conn)
    items = build_items()
    n = 0
    for it in items:
        conn.execute(
            "INSERT OR REPLACE INTO benchmark_items(item_key, layer, dimension, "
            "scenario_id, variant, prompt, rubric_json, holdout) VALUES (?,?,?,?,?,?,?,?)",
            (it["item_key"], it["layer"], it["dimension"], it["scenario_id"],
             it["variant"], it["prompt"],
             json.dumps(it["rubric_json"], ensure_ascii=False), it["holdout"]))
        n += 1
    conn.commit()
    layers = conn.execute("SELECT layer, COUNT(*) FROM benchmark_items GROUP BY layer").fetchall()
    print(f"inserted {n} items:")
    for layer, cnt in layers:
        print(f"  {layer}: {cnt}")


if __name__ == "__main__":
    main()
