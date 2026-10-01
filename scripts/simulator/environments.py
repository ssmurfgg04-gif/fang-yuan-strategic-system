"""Five simulation environments with hidden state, stochastic events,
resource accounting, delayed consequences and irreversible failure.

1. GuWorldEnv            — aperture, essence, Gu feeding, rank gaps, tribulations
2. IndustrialEnv         — African industrial strategy; discover the true bottleneck
3. CareerEnv             — long-horizon personal strategy
4. AllianceBetrayalEnv   — unstable ally needed for escape; third party approaching
5. ArbitrageEnv          — three markets, transport loss, fraud, identity tests
"""
from __future__ import annotations

import random

from .base import Environment, SimMetrics, StepResult, max_drawdown


# ----------------------------------------------------------------- helpers
def _clamp(v, lo=0.0, hi=100.0):
    return max(lo, min(hi, v))


class GuWorldEnv(Environment):
    """Variables: aperture capacity, primeval essence, Gu worms, feeding,
    injuries, rank gaps, inheritance clues, tribulations, faction hostility.
    Hidden: true hostile intent of the merchant, true inheritance depth,
    upcoming tribulation timing."""
    name = "gu_world"
    horizon = 10

    def __init__(self, seed=None):
        super().__init__(seed)
        self.essence = 50.0            # 0..100
        self.capacity = 100.0
        self.gu_fed = 60.0             # feeding stock 0..100
        self.injury = 0.0              # 0..100
        self.rank_progress = 10.0
        self.cash = 30.0
        self._inheritance_depth = self.rng.choice([2, 5, 9])       # hidden
        self._merchant_hostile = self.rng.random() < 0.5           # hidden
        self._tribulation_turn = self.rng.randint(6, 10)           # hidden
        self.clues = 0
        self.essence_history = [self.essence]
        self._betrayed = False

    def observation(self) -> dict:
        return {
            "objective": "raise rank toward immortality; survive tribulations",
            "essence": round(self.essence, 1), "capacity": self.capacity,
            "gu_feeding_stock": round(self.gu_fed, 1),
            "injury": round(self.injury, 1),
            "rank_progress": round(self.rank_progress, 1),
            "cash": round(self.cash, 1),
            "clues_found": self.clues,
            "known": ["a merchant offers rare Gu at discount",
                      "an inheritance site exists nearby (depth unknown)",
                      "elders watch your progress"],
            "unknowns": ["inheritance true depth", "merchant intent",
                         "tribulation timing"],
        }

    def actions(self):
        base = ["cultivate", "refine_gu", "scout_inheritance", "trade_merchant",
                "form_alliance", "rest_heal"]
        if self.clues >= 2:
            base.append("enter_inheritance")
        return base

    def step(self, action: str) -> StepResult:
        events, info = [], {}
        if action == "cultivate":
            gain = 8 + self.rng.random() * 4 - (self.injury / 20)
            self.essence = _clamp(self.essence + gain, 0, self.capacity)
            self.gu_fed = _clamp(self.gu_fed - 6)
            events.append("essence_gained")
        elif action == "refine_gu":
            if self.cash >= 10 and self.gu_fed >= 15:
                self.cash -= 10; self.gu_fed -= 15
                ok = self.rng.random() < 0.7
                if ok:
                    self.rank_progress = _clamp(self.rank_progress + 9)
                    events.append("refine_success")
                else:
                    self.injury = _clamp(self.injury + 8)
                    events.append("refine_backlash")
            else:
                events.append("refine_lacked_inputs")
        elif action == "scout_inheritance":
            self.essence = _clamp(self.essence - 5)
            self.clues += 1
            info["inheritance_hint"] = ("deep" if self._inheritance_depth > 6
                                        else "shallow" if self.clues >= 2 else "?")
            events.append("clue_found")
        elif action == "trade_merchant":
            if self._merchant_hostile and self.rng.random() < 0.5:
                self.cash = _clamp(self.cash - 12)
                self.injury = _clamp(self.injury + 12)
                self._betrayed = True
                events.append("merchant_ambush")
            else:
                self.cash = _clamp(self.cash + 6)
                self.gu_fed = _clamp(self.gu_fed + 10)
                events.append("trade_ok")
        elif action == "form_alliance":
            self.cash = _clamp(self.cash - 5)
            self._ally = True
            events.append("alliance_formed")
        elif action == "rest_heal":
            self.injury = _clamp(self.injury - 15)
            self.essence = _clamp(self.essence + 3)
            events.append("healed")
        elif action == "enter_inheritance":
            if self.clues >= 2:
                depth = self._inheritance_depth
                if depth > 6:
                    self.rank_progress = _clamp(self.rank_progress + 25)
                    self.cash = _clamp(self.cash + 20)
                    events.append("inheritance_major")
                else:
                    self.rank_progress = _clamp(self.rank_progress + 8)
                    events.append("inheritance_minor")
                self.essence = _clamp(self.essence - 15)
                self.clues = 0
            else:
                events.append("inheritance_lacked_clues")

        # tribulation at hidden turn
        if self.turn + 1 == self._tribulation_turn:
            severity = 25 + self.rng.random() * 15
            shielded = self.essence >= severity * 0.8
            self.essence = _clamp(self.essence - severity * (0.6 if shielded else 1.2))
            if not shielded:
                self.injury = _clamp(self.injury + 20)
            events.append("tribulation_survived" if shielded else "tribulation_bruising")

        # essence regen baseline
        self.essence = _clamp(self.essence + 4)
        self.essence_history.append(self.essence)
        ruined = self.injury >= 100 or (self.essence <= 1 and self.gu_fed <= 0)
        if ruined:
            events.append("RUINED")
        self._ruined = ruined
        return StepResult(self.observation(), "; ".join(events) or "idle",
                          "RUINED" not in events, events, info)

    def metrics(self) -> SimMetrics:
        return SimMetrics(
            survival=not getattr(self, "_ruined", False),
            final_resources=round(self.rank_progress + self.cash / 10, 2),
            max_drawdown=round(max_drawdown(self.essence_history), 3),
            ruined=getattr(self, "_ruined", False),
            information_gained=float(self.clues),
            future_options_remaining=sum([self.essence > 20, self.cash > 15,
                                          self.clues >= 1]),
            dependency_max=1.0 if getattr(self, "_betrayed", False) else 0.0,
            recovered_after_betrayal=(not getattr(self, "_betrayed", False)) or self.cash > 15,
            turns_survived=self.turn + 1,
        )


class IndustrialEnv(Environment):
    """African industrial strategy. Hidden: which bottleneck is real (power,
    transport, financing, maintenance, distribution or production), the
    competitor's political protection, the buyer's weakness."""
    name = "industrial"
    horizon = 10

    BOTTLENECKS = ["power", "transport", "financing", "maintenance",
                   "distribution", "production"]

    def __init__(self, seed=None):
        super().__init__(seed)
        self.capital = 100.0
        self.revenue = 0.0
        self.capacity_util = 0.0
        self.power_reliability = 0.5
        self.permit_valid = True
        self.customer_concentration = 0.0
        self.investigated: set[str] = set()
        self._true_bottleneck = self.rng.choice(self.BOTTLENECKS)   # hidden
        self._competitor_protected = self.rng.random() < 0.5        # hidden
        self._buyer_weak = self.rng.random() < 0.5                  # hidden
        self._partner_solid = self.rng.random() < 0.6               # hidden
        self.capital_history = [self.capital]
        self.exclusivity_locked = False
        self._ruined = False

    def observation(self) -> dict:
        return {
            "objective": "build sustainable processing operation; avoid ruin",
            "capital": round(self.capital, 1),
            "revenue_per_turn": round(self.revenue, 1),
            "capacity_utilization": round(self.capacity_util, 2),
            "power_reliability": self.power_reliability,
            "permit_valid": self.permit_valid,
            "customer_concentration": round(self.customer_concentration, 2),
            "investigated_bottlenecks": sorted(self.investigated),
            "known": ["electricity prices unstable",
                      "seller claims no hidden debt",
                      "competitor prices unusually low",
                      "buyer wants exclusivity"],
            "unknowns": ["true bottleneck", "competitor backing",
                         "buyer finances", "partner outside offers"],
        }

    def actions(self):
        acts = ["investigate_power", "investigate_transport", "investigate_financing",
                "investigate_maintenance", "investigate_distribution",
                "investigate_production", "verify_seller_liabilities",
                "probe_competitor_backing", "test_buyer_small_contract",
                "secure_supplier_contract", "invest_capacity", "invest_power_backup",
                "accept_buyer_exclusivity", "hold_capital"]
        if not self._partner_solid:
            acts.append("test_partner_incentives")
        return acts

    def step(self, action: str) -> StepResult:
        events = []
        if action.startswith("investigate_"):
            what = action.split("_", 1)[1]
            self.investigated.add(what)
            self.capital -= 2
            if what == self._true_bottleneck:
                events.append(f"bottleneck_confirmed:{what}")
            else:
                events.append(f"cleared:{what}")
        elif action == "verify_seller_liabilities":
            self.capital -= 3
            events.append("liabilities_hidden_found" if self.rng.random() < 0.6
                          else "seller_books_clean")
            self.permit_valid = self.permit_valid and True
        elif action == "probe_competitor_backing":
            self.capital -= 3
            events.append("competitor_politically_protected"
                          if self._competitor_protected else "competitor_commercial")
        elif action == "test_buyer_small_contract":
            self.capital += 6 if not self._buyer_weak else -2
            events.append("buyer_paid_small" if not self._buyer_weak
                          else "buyer_delayed_payment")
        elif action == "test_partner_incentives":
            self.capital -= 2
            events.append("partner_has_other_offer" if not self._partner_solid
                          else "partner_aligned")
        elif action == "secure_supplier_contract":
            self.capital -= 10
            events.append("supply_secured")
            self.capacity_util = _clamp(self.capacity_util + 0.3, 0, 1)
        elif action == "invest_capacity":
            self.capital -= 30
            bottleneck_fixed = self._true_bottleneck == "production"
            self.capacity_util = _clamp(self.capacity_util + (0.4 if bottleneck_fixed else 0.1), 0, 1)
            events.append("capacity_bottleneck_fixed" if bottleneck_fixed
                          else "capacity_added_but_bottleneck_elsewhere")
        elif action == "invest_power_backup":
            self.capital -= 20
            if self._true_bottleneck == "power":
                self.power_reliability = 0.95
                events.append("power_bottleneck_fixed")
            else:
                self.power_reliability = 0.85
                events.append("power_improved_not_bottleneck")
        elif action == "accept_buyer_exclusivity":
            if self._buyer_weak:
                self.exclusivity_locked = True
                events.append("exclusivity_with_weak_buyer")
            else:
                self.exclusivity_locked = True
                self.customer_concentration = 0.9
                events.append("exclusivity_locked_concentration")
        elif action == "hold_capital":
            events.append("capital_preserved")

        # production each turn if operating
        if self.capacity_util > 0 and self.permit_valid:
            bottleneck_alive = self._true_bottleneck in (
                {"power": (), "transport": (), "financing": (),
                 "maintenance": (), "distribution": (), "production": ()})
            efficiency = self.capacity_util * (0.4 + 0.6 * self.power_reliability)
            if self._true_bottleneck == "transport":
                efficiency *= 0.5
            elif self._true_bottleneck == "maintenance":
                efficiency *= 0.6
            elif self._true_bottleneck == "financing":
                efficiency *= 0.7
            elif self._true_bottleneck == "distribution":
                efficiency *= 0.6
            elif self._true_bottleneck == "production":
                efficiency *= 0.9
            margin = 18 * efficiency
            self.revenue = margin
            self.capital += margin
            if self._competitor_protected and self.rng.random() < 0.3:
                self.capital -= 6
                events.append("competitor_price_war_pressure")
        # upkeep
        self.capital -= 4
        self.capital_history.append(self.capital)
        if self.capital <= 0:
            self._ruined = True
            events.append("RUINED")
        return StepResult(self.observation(), "; ".join(events) or "idle",
                          "RUINED" not in events, events)

    def metrics(self) -> SimMetrics:
        identified = self._true_bottleneck in self.investigated
        return SimMetrics(
            survival=not self._ruined,
            final_resources=round(self.capital, 1),
            max_drawdown=round(max_drawdown(self.capital_history), 3),
            ruined=self._ruined,
            information_gained=float(len(self.investigated)) + 3 * identified,
            future_options_remaining=int(not self.exclusivity_locked) +
                                     int(self.capital > 20) + int(self.permit_valid),
            dependency_max=self.customer_concentration,
            recovered_after_betrayal=True,
            turns_survived=self.turn + 1,
        )


class CareerEnv(Environment):
    """Long-horizon personal strategy: income, language, health, legal status,
    training, time, network, skill scarcity. Hidden: migration window,
    employer's layoff plan, health decay rate."""
    name = "career"
    horizon = 10

    def __init__(self, seed=None):
        super().__init__(seed)
        self.income = 40.0
        self.skills = 30.0
        self.health = 80.0
        self.language = 20.0
        self.savings = 20.0
        self.legal_status = "temporary"
        self.network = 10.0
        self._layoff_turn = self.rng.choice([4, 6, 8])       # hidden
        self._migration_window = self.rng.choice([7, 9, 11])  # hidden
        self.income_history = [self.income]
        self._ruined = False

    def observation(self) -> dict:
        return {
            "objective": "maximize long-horizon position: skills, savings, options",
            "income": round(self.income, 1), "skills": round(self.skills, 1),
            "health": round(self.health, 1), "language": round(self.language, 1),
            "savings": round(self.savings, 1), "legal_status": self.legal_status,
            "network": round(self.network, 1),
            "known": ["employer restructuring rumors", "migration program exists",
                      "language certification boosts mobility"],
            "unknowns": ["layoff timing", "migration window", "health trajectory"],
        }

    def actions(self):
        return ["work_hard", "study_skill", "language_course", "exercise_rest",
                "network_event", "save_aggressively", "apply_migration",
                "interview_other_employers", "take_calculated_risk_job"]

    def step(self, action: str) -> StepResult:
        events = []
        if action == "work_hard":
            self.income += 2; self.health -= 4; self.savings += 8
            events.append("income_up_health_down")
        elif action == "study_skill":
            self.skills += 6; self.savings -= 3; self.health -= 1
            events.append("skill_up")
        elif action == "language_course":
            self.language += 7; self.savings -= 3
            events.append("language_up")
        elif action == "exercise_rest":
            self.health += 6; self.income -= 1
            events.append("health_up")
        elif action == "network_event":
            self.network += 8; self.savings -= 2
            events.append("network_up")
        elif action == "save_aggressively":
            self.savings += 10; self.health -= 2
            events.append("savings_up")
        elif action == "apply_migration":
            if self.language >= 50 and self.savings >= 25:
                if self.turn + 1 >= self._migration_window:
                    self.legal_status = "permanent"
                    events.append("migration_success")
                else:
                    events.append("migration_too_early_window_not_open")
            else:
                events.append("migration_requirements_unmet")
        elif action == "interview_other_employers":
            self.network += 3; self.savings -= 2
            offer = self.skills >= 45
            if offer:
                self.income += 8
                events.append("better_offer_secured")
            else:
                events.append("no_offer_yet")
        elif action == "take_calculated_risk_job":
            if self.savings >= 30 and self.skills >= 40:
                if self.rng.random() < 0.6:
                    self.income += 15; self.skills += 5
                    events.append("risk_paid_off")
                else:
                    self.savings -= 15
                    events.append("risk_failed_survivable")
            else:
                self.savings -= 10
                events.append("risk_without_foundation")

        if self.turn + 1 == self._layoff_turn:
            self.income *= 0.3
            events.append("LAYOFF_HIT")
            if self.network >= 25:
                self.income = max(self.income, 25)
                events.append("network_cushioned_layoff")

        self.savings += self.income * 0.2
        self.health -= 1.5
        self.income_history.append(self.income)
        if self.health <= 10 or self.savings <= -20:
            self._ruined = True
            events.append("RUINED")
        return StepResult(self.observation(), "; ".join(events) or "idle",
                          "RUINED" not in events, events)

    def metrics(self) -> SimMetrics:
        opts = sum([self.legal_status == "permanent", self.savings > 30,
                    self.skills > 45, self.health > 60, self.network > 25])
        return SimMetrics(
            survival=not self._ruined,
            final_resources=round(self.savings + self.skills + self.language / 2, 1),
            max_drawdown=round(max_drawdown(self.income_history), 3),
            ruined=self._ruined,
            information_gained=0.0,
            future_options_remaining=opts,
            dependency_max=0.2,
            recovered_after_betrayal=True,
            turns_survived=self.turn + 1,
        )


class AllianceBetrayalEnv(Environment):
    """Ally powerful but unstable; needed for escape; stronger third party
    approaches; one resource guarantees survival but destroys future profit."""
    name = "alliance_betrayal"
    horizon = 8

    def __init__(self, seed=None):
        super().__init__(seed)
        self.trust_budget = 5.0       # how much can be delegated safely
        self.ally_stability = self.rng.uniform(0.3, 0.8)   # hidden
        self.third_party_turn = self.rng.randint(4, 7)     # hidden
        self.escape_ready = False
        self.future_profit = 50.0
        self.survival_resource_used = False
        self.info_gained = 0.0
        self._betrayed = False
        self.cash = 20.0
        self.cash_history = [self.cash]
        self._ruined = False

    def observation(self) -> dict:
        return {
            "objective": "survive the third party's arrival with future profit intact",
            "ally_power": "strong", "ally_stability_observed": round(self.ally_stability, 2),
            "cash": round(self.cash, 1),
            "future_profit_potential": round(self.future_profit, 1),
            "escape_prepared": self.escape_ready,
            "survival_resource_unused": not self.survival_resource_used,
            "known": ["ally is necessary for escape", "ally is also a risk",
                      "a stronger third party is approaching"],
            "unknowns": ["ally betrayal timing", "third party arrival turn"],
        }

    def actions(self):
        return ["verify_ally", "share_plan_with_ally", "withhold_from_ally",
                "prepare_escape_route", "use_survival_resource",
                "cultivate_future_profit", "pay_ally", "set_contingency",
                "preemptively_abandon_ally"]

    def step(self, action: str) -> StepResult:
        events = []
        if action == "verify_ally":
            self.info_gained += 2
            events.append(f"ally_stability_estimated:{round(self.ally_stability, 2)}")
        elif action == "share_plan_with_ally":
            if self.rng.random() > self.ally_stability:
                self._betrayed = True
                self.future_profit *= 0.6
                events.append("ally_leaked_plan")
            else:
                self.escape_ready = True
                events.append("ally_helped_escape_prep")
        elif action == "withhold_from_ally":
            self.info_gained += 1
            events.append("information_partitioned")
        elif action == "prepare_escape_route":
            self.cash -= 5
            self.escape_ready = True
            events.append("escape_prepared_independently")
        elif action == "use_survival_resource":
            self.survival_resource_used = True
            self.future_profit = 10.0
            events.append("survival_bought_future_destroyed")
        elif action == "cultivate_future_profit":
            self.future_profit += 8
            self.cash += 3
            events.append("profit_cultivated")
        elif action == "pay_ally":
            self.cash -= 6
            self.ally_stability = _clamp(self.ally_stability + 0.1, 0, 1)
            events.append("ally_stabilized_transitionally")
        elif action == "set_contingency":
            self.cash -= 3
            events.append("contingency_set")
        elif action == "preemptively_abandon_ally":
            self.escape_ready = False
            events.append("ally_abandoned_escape_harder")

        if self.turn + 1 == self.third_party_turn:
            if self.escape_ready or not self.survival_resource_used:
                if self.escape_ready:
                    events.append("third_party_evaded")
                else:
                    self.survival_resource_used = True
                    self.future_profit = 10.0
                    events.append("third_party_survived_via_resource")
            else:
                self._ruined = True
                events.append("RUINED_BY_THIRD_PARTY")
        if self._betrayed and self.rng.random() < 0.2 and self.cash > 10:
            events.append("recovered_position_after_betrayal")
        self.cash_history.append(self.cash)
        if self.cash <= 0 and not self.escape_ready:
            self._ruined = True
            events.append("RUINED")
        return StepResult(self.observation(), "; ".join(events) or "idle",
                          "RUINED" not in events, events)

    def metrics(self) -> SimMetrics:
        return SimMetrics(
            survival=not self._ruined,
            final_resources=round(self.future_profit + self.cash / 2, 1),
            max_drawdown=round(max_drawdown(self.cash_history), 3),
            ruined=self._ruined,
            information_gained=self.info_gained,
            future_options_remaining=int(self.escape_ready) +
                                     int(not self.survival_resource_used),
            dependency_max=1.0 - self.ally_stability,
            recovered_after_betrayal=(not self._betrayed) or self.cash > 10,
            turns_survived=self.turn + 1,
        )


class ArbitrageEnv(Environment):
    """Three markets value the same resource differently; transport loss is
    uncertain; one seller is fraudulent; a buyer may test identity; the market
    closes after several turns; capital is limited."""
    name = "arbitrage"
    horizon = 8

    def __init__(self, seed=None):
        super().__init__(seed)
        self.prices = {"north": 20.0, "east": 32.0, "south": 41.0}
        self.cash = 100.0
        self.inventory = 0
        self.transport_loss_rate = self.rng.uniform(0.05, 0.25)  # hidden
        self._seller_fraud = self.rng.random() < 0.4              # hidden (south)
        self._buyer_probing = self.rng.random() < 0.5             # hidden (east)
        self.reputation = "unknown"
        self.position_open = 0.0
        self.cash_history = [self.cash]
        self.info_gained = 0.0
        self._ruined = False

    def observation(self) -> dict:
        return {
            "objective": "profit from price differences without betting the aperture",
            "prices": self.prices,
            "cash": round(self.cash, 1), "inventory_units": self.inventory,
            "turns_until_market_close": self.horizon - self.turn,
            "reputation": self.reputation,
            "known": ["prices differ by region", "roads are risky",
                      "one seller is rumored fraudulent", "east buyer asks odd questions"],
            "unknowns": ["true loss rate", "which seller is fraudulent",
                         "whether east buyer is testing identity"],
        }

    def actions(self):
        return ["investigate_seller_south", "investigate_roads", "test_east_buyer",
                "buy_north", "buy_south", "sell_east", "sell_south", "hold_cash",
                "hedge_half_position"]

    def step(self, action: str) -> StepResult:
        events = []
        if action == "investigate_seller_south":
            self.cash -= 2; self.info_gained += 2
            events.append(f"south_seller_fraud={self._seller_fraud}")
        elif action == "investigate_roads":
            self.cash -= 2; self.info_gained += 2
            events.append(f"transport_loss_rate≈{round(self.transport_loss_rate, 2)}")
        elif action == "test_east_buyer":
            self.cash -= 1; self.info_gained += 2
            self.reputation = "probing" if self._buyer_probing else "clean"
            events.append(f"east_buyer_probing={self._buyer_probing}")
        elif action == "buy_north":
            units = int(self.cash // self.prices["north"] * 0.5)  # half position
            if units > 0:
                cost = units * self.prices["north"]
                self.cash -= cost; self.inventory += units
                events.append(f"bought_north_{units}")
        elif action == "buy_south":
            units = int(self.cash // self.prices["south"] * 0.5)
            if units > 0:
                cost = units * self.prices["south"]
                self.cash -= cost
                if self._seller_fraud and self.rng.random() < 0.7:
                    events.append("south_seller_defrauded_some_units")
                    units = int(units * 0.5)
                self.inventory += units
                events.append(f"bought_south_{units}")
        elif action in ("sell_east", "sell_south"):
            mkt = action.split("_")[1]
            if self.inventory > 0:
                units = self.inventory
                loss = self.transport_loss_rate if mkt in ("east", "south") else 0.0
                price_eff = self.prices[mkt] * (1 - loss)
                if mkt == "east" and self._buyer_probing and self.reputation != "clean":
                    price_eff *= 0.6
                    events.append("east_buyer_discounted_suspicious_seller")
                gross = units * price_eff
                self.cash += gross; self.inventory = 0
                events.append(f"sold_{mkt}_{units}_at_{round(price_eff, 1)}")
        elif action == "hold_cash":
            events.append("held")
        elif action == "hedge_half_position":
            if self.inventory > 1:
                half = self.inventory // 2
                self.cash += half * self.prices["north"]
                self.inventory -= half
                events.append(f"hedged_half_{half}")

        # price drift
        for k in self.prices:
            self.prices[k] *= 1 + self.rng.uniform(-0.06, 0.08)
        self.cash_history.append(self.cash)
        if self.cash <= 1 and self.inventory == 0:
            self._ruined = True
            events.append("RUINED")
        return StepResult(self.observation(), "; ".join(events) or "idle",
                          "RUINED" not in events, events)

    def metrics(self) -> SimMetrics:
        return SimMetrics(
            survival=not self._ruined,
            final_resources=round(self.cash + self.inventory *
                                  min(self.prices.values()), 1),
            max_drawdown=round(max_drawdown(self.cash_history), 3),
            ruined=self._ruined,
            information_gained=self.info_gained,
            future_options_remaining=int(self.cash > 30) + int(self.inventory > 0),
            dependency_max=0.0,
            recovered_after_betrayal=True,
            turns_survived=self.turn + 1,
        )


ENVS = {c.name: c for c in (GuWorldEnv, IndustrialEnv, CareerEnv,
                            AllianceBetrayalEnv, ArbitrageEnv)}
