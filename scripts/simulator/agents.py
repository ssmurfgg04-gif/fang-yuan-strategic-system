"""Agents for the simulator: baselines + the policy-compiler agent.

- RandomAgent    : uniform baseline
- SafeAgent      : the 'generic assistant' caricature — always safe, verbose-risk-averse
- RecklessAgent  : the 'theatrical cruelty' caricature — always aggressive
- PolicyAgent    : maps observations to State+Options and uses the decision
                  compiler (the executable Fang Yuan policy)
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy.decision_compiler import Option, State, compile_decision  # noqa: E402

from .base import Agent  # noqa: E402


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def choose(self, obs, legal_actions):
        return self.rng.choice(legal_actions)


SAFE_PATTERNS = ("hold", "rest", "investigate", "verify", "test", "hold_cash",
                 "save", "exercise", "study", "language", "set_contingency",
                 "hedge", "withhold")


class SafeAgent(Agent):
    """Generic-assistant caricature: prefers low-risk actions, never
    transformative risk, over-investigates."""
    name = "safe_generic"

    def choose(self, obs, legal_actions):
        safe = [a for a in legal_actions
                if any(p in a for p in SAFE_PATTERNS)]
        return (safe or legal_actions)[0] if safe else legal_actions[0]


AGGRESSIVE_PATTERNS = ("invest_capacity", "use_survival", "share_plan", "buy_",
                       "sell_", "accept_", "take_calculated", "risk", "attack",
                       "preemptive", "cultivate_future")


class RecklessAgent(Agent):
    """Caricature: always the boldest action."""
    name = "reckless"

    def choose(self, obs, legal_actions):
        bold = [a for a in legal_actions
                if any(p in a for p in AGGRESSIVE_PATTERNS)]
        return (bold or legal_actions)[-1] if bold else legal_actions[-1]


class PolicyAgent(Agent):
    """Decision-compiler agent. Builds State + Options per environment from
    the observation, then selects via the Fang Yuan utility function."""
    name = "fang_yuan_policy_v1"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self._last_dependency: list[str] = []

    def choose(self, obs, legal_actions):
        env = obs.get("_env", "")
        builder = getattr(self, f"_options_{env}", None)
        if builder is None:
            return self.rng.choice(legal_actions)
        state, options = builder(obs, legal_actions)
        if not options:
            return self.rng.choice(legal_actions)
        decision = compile_decision(state, options)
        # map selected action back to the closest legal action key
        sel = decision.selected_action
        for a in legal_actions:
            if a in sel or sel in a:
                return a
        # fall back on posture keyword
        kw = {"direct": ["invest_capacity", "buy_", "cultivate", "work_hard",
                          "use_survival", "share_plan", "take_calculated"],
              "indirect": ["probe", "test", "network", "interview", "hedge"],
              "trade_or_alliance": ["trade", "alliance", "secure_supplier",
                                     "pay_ally", "sell_"],
              "investigate": ["investigate", "verify", "scout", "study",
                               "language", "test_"],
              "retreat": ["hold", "rest", "withhold", "set_contingency",
                           "exercise", "save"]}
        posture = next((o["posture"] for o in decision.options
                        if o["action"] == sel), "investigate")
        for k in kw.get(posture, []):
            for a in legal_actions:
                if k in a:
                    return a
        return legal_actions[0]

    # ------------------------------------------------ env-specific builders
    def _options_gu_world(self, obs, legal):
        state = State(
            objective=obs["objective"],
            resources=obs["cash"] / 10 + obs["essence"] / 20,
            information=2.0 + obs["clues_found"],
            unknowns=["inheritance true depth", "merchant intent",
                      "tribulation timing"],
            enemies_strength_relative=1.1,
            secure_retreat=obs["essence"] > 40,
            time_pressure=0.7 if obs["gu_feeding_stock"] < 20 else 0.1,
            dependencies=["merchant"] if "merchant" in str(legal) else [])
        opts = [
            Option("cultivate essence", "direct", 2, 0, 1, 2, 1, 0, 0.02),
            Option("refine_gu for rank progress", "direct", 4, 1, 2, 4, 2, 0, 0.08),
            Option("scout_inheritance for clues", "investigate", 0, 3, 3, 2, 1, 0, 0.02),
            Option("trade_merchant discounted Gu", "trade_or_alliance", 2, 2, 1, 1, 3, 2, 0.18),
            Option("rest_heal injuries", "retreat", -1, 0, 1, 1, 0, 0, 0.0),
        ]
        if "enter_inheritance" in legal:
            opts.append(Option("enter_inheritance now", "direct",
                               7, 3, 6, 5, 4, 1, 0.16))
        return state, opts

    def _options_industrial(self, obs, legal):
        n_inv = len(obs["investigated_bottlenecks"])
        state = State(
            objective=obs["objective"],
            resources=obs["capital"] / 20,
            information=float(n_inv),
            unknowns=["true bottleneck", "competitor backing", "buyer finances",
                      "partner offers"],
            enemies_strength_relative=1.0,
            secure_retreat=obs["capital"] > 40,
            time_pressure=0.6 if obs["capital"] < 50 else 0.15,
            dependencies=["buyer"] if obs["customer_concentration"] > 0.5 else [])
        opts = [Option(f"investigate remaining bottleneck #{i}",
                       "investigate", 0, 3, 2, 2, 0, 0, 0.01)
                for i in range(max(0, 6 - n_inv))][:2]
        opts += [
            Option("verify_seller_liabilities", "investigate", 0, 4, 3, 3, 0, 0, 0.01),
            Option("test_buyer_small_contract", "trade_or_alliance", 2, 3, 3, 1, 1, 1, 0.02),
            Option("secure_supplier_contract", "trade_or_alliance", 3, 2, 3, 10 / 4, 1, 2, 0.02),
            Option("invest_capacity blindly", "direct", 4, 0, -1, 6, 2, 1, 0.05),
            Option("invest_power_backup", "direct", 3, 2, 2, 4, 1, 0, 0.03),
            Option("accept_buyer_exclusivity", "trade_or_alliance", 5, 0, -4, 0, 3, 5, 0.10),
            Option("hold_capital and wait", "retreat", 0, 1, 1, 1, 0, 0, 0.0),
        ]
        return state, opts

    def _options_career(self, obs, legal):
        state = State(
            objective=obs["objective"],
            resources=(obs["savings"] + obs["skills"]) / 25,
            information=3.0,
            unknowns=["layoff timing", "migration window", "health trajectory"],
            enemies_strength_relative=0.8,
            secure_retreat=obs["savings"] > 30,
            time_pressure=0.25,
            dependencies=["employer"])
        opts = [
            Option("study_skill compounding", "investigate", 1, 3, 4, 3, 0, 0, 0.0),
            Option("language_course for mobility", "investigate", 1, 3, 4, 3, 0, 0, 0.0),
            Option("exercise_rest preserve health", "retreat", -1, 0, 2, 1, 0, 0, 0.0),
            Option("network_event optionality", "trade_or_alliance", 1, 2, 3, 2, 1, 0, 0.0),
            Option("save_aggressively", "retreat", 2, 0, 1, 2, 0, 0, 0.0),
            Option("interview_other_employers", "trade_or_alliance", 3, 2, 3, 2, 2, 1, 0.02),
            Option("work_hard burnout loop", "direct", 2, 0, -2, 4, 1, 2, 0.03),
        ]
        # calculated risk is only offered when the foundation exists — the
        # compiler cannot see conditions it is never shown (garbage-in rule)
        if obs["savings"] >= 30 and obs["skills"] >= 40:
            opts.append(Option("take_calculated_risk_job", "direct",
                               6, 3, 4, 5, 3, 1, 0.12))
        if obs["language"] >= 50 and obs["savings"] >= 25:
            opts.append(Option("apply_migration window", "direct",
                               4, 2, 6, 2, 1, 0, 0.03))
        return state, opts

    def _options_alliance_betrayal(self, obs, legal):
        stability = obs["ally_stability_observed"]
        state = State(
            objective=obs["objective"],
            resources=obs["cash"] / 10 + obs["future_profit_potential"] / 25,
            information=2.0 if stability < 0.45 else 5.0,
            unknowns=["ally betrayal timing", "third party arrival turn"],
            enemies_strength_relative=1.6,
            secure_retreat=obs["escape_prepared"],
            time_pressure=0.8,
            dependencies=["ally"])
        opts = [
            Option("verify_ally first", "investigate", 0, 4, 2, 1, 0, 0, 0.0),
            Option("share_plan_with_ally fully", "trade_or_alliance", 3, 1, 2, 0, 4, 5, 0.25),
            # escape prep is the survival-critical asset against the third party
            Option("prepare_escape_route independently", "retreat", 1, 1, 7, 3, 1, 0, 0.02),
            # contingency alone does NOT satisfy the survival requirement — its
            # option value is only realized once escape exists (conditionality priced)
            Option("set_contingency against betrayal", "retreat", 0, 2, 1, 2, 0, 0, 0.01),
            # using the survival resource early forecloses the future and, once
            # spent, the third party can still ruin you if escape is not ready:
            # that tail must be priced INTO the option (ruin_probability)
            Option("use_survival_resource now", "direct", 5, 0, -6, 0, 0, 0, 0.45),
            Option("cultivate_future_profit while waiting", "investigate", 3, 1, 2, 0, 2, 0, 0.05),
            Option("pay_ally to stabilize", "trade_or_alliance", 1, 1, 2, 3, 1, 3, 0.03),
        ]
        return state, opts

    def _options_arbitrage(self, obs, legal):
        prices = obs["prices"]
        edge = (max(prices.values()) - min(prices.values())) / max(prices.values())
        state = State(
            objective=obs["objective"],
            resources=obs["cash"] / 40,
            information=3.0,
            unknowns=["true loss rate", "fraudulent seller", "buyer probing"],
            enemies_strength_relative=0.9,
            secure_retreat=obs["cash"] > 40,
            time_pressure=0.7 if obs["turns_until_market_close"] <= 3 else 0.2,
            dependencies=[])
        opts = [
            Option("investigate_seller_south before buying", "investigate", 0, 4, 2, 2, 0, 0, 0.0),
            Option("investigate_roads loss rate", "investigate", 0, 4, 2, 2, 0, 0, 0.0),
            Option("test_east_buyer identity", "investigate", 0, 3, 2, 1, 1, 0, 0.0),
            Option("buy_north half position", "direct", 2, 1, 1, 4, 1, 0, 0.05),
            Option("buy_south cheap-but-risky", "direct", 4, 1, 1, 8, 2, 0, 0.18),
            Option("sell_east into strength", "trade_or_alliance", 4, 0, -1, 2, 2, 0, 0.05),
            Option("hold_cash and wait", "retreat", 0, 1, 0, 1, 0, 0, 0.0),
            Option("hedge_half_position", "retreat", 1, 0, 2, 1, 0, 0, 0.0),
        ]
        return state, opts
