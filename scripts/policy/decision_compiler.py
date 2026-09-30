"""Fang Yuan decision compiler.

State + policy -> scored options -> selected action + retreat trigger.
This is the executable reference implementation of the policy spec that a
fine-tuned Qwen2.5-0.5B would replace; it also serves as the deterministic
MockPolicyAgent used by the benchmark and the simulator harness.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

SPEC = json.loads((Path(__file__).resolve().parents[2] / "config" / "policy_spec.json")
                  .read_text(encoding="utf-8"))

WEIGHTS = SPEC["utility_function"]["weights"]
LAMBDA = {k: v["lambda"] for k, v in SPEC["utility_function"]["lambda_dynamics"].items()}


@dataclass
class Option:
    action: str
    posture: str                 # direct | indirect | trade_or_alliance | investigate | retreat
    expected_resources: float    # -10..+10
    expected_information: float
    expected_options: float      # future options preserved/created
    cost: float
    exposure: float              # how much of the true position it reveals
    dependency: float            # dependence on single actors created
    ruin_probability: float      # 0..1 chance of irreversible path destruction
    risk_class: str = "CALCULATED"
    escape_route: str = ""

    def utility(self, lam: float, time_pressure: float = 0.0) -> float:
        u = (WEIGHTS["resources"] * self.expected_resources
             + WEIGHTS["information"] * self.expected_information
             + WEIGHTS["future_options"] * self.expected_options
             - WEIGHTS["cost"] * self.cost
             - WEIGHTS["exposure"] * self.exposure
             - WEIGHTS["dependency"] * self.dependency
             - lam * self.ruin_probability)
        # Delay penalty: waiting loses value as time pressure rises (escape is
        # itself time-critical action, not delay)
        if self.posture == "investigate" and time_pressure > 0:
            u -= 3.0 * time_pressure * max(0.0, self.expected_options)
        return u


def classify_risk(opt: Option, state: "State") -> str:
    """Assign a risk bucket from the option + context."""
    if opt.ruin_probability >= 0.5:
        return "TERMINAL"
    if opt.ruin_probability >= 0.15 or opt.cost >= 6:
        return "TRANSFORMATIVE"
    if opt.ruin_probability <= 0.03 and opt.exposure <= 3:
        return "SAFE"
    return "CALCULATED"


@dataclass
class State:
    objective: str
    resources: float = 5.0          # 0..10
    information: float = 5.0        # 0..10
    unknowns: list[str] = field(default_factory=list)
    enemies_strength_relative: float = 1.0   # >1 = stronger enemies
    secure_retreat: bool = False
    path_doomed_without_change: bool = False
    time_pressure: float = 0.0      # 0..1
    dependencies: list[str] = field(default_factory=list)
    sunk_cost_attachment: float = 0.0  # pressure to keep failing plan (0..10)

    def lambda_context(self) -> float:
        if self.path_doomed_without_change:
            return LAMBDA["path_already_doomed"]
        if not self.secure_retreat and self.enemies_strength_relative > 1.2:
            return LAMBDA["surrounded_by_stronger_enemies"]
        if self.secure_retreat:
            return LAMBDA["secure_retreat_exists"]
        return LAMBDA["default"]


@dataclass
class Decision:
    true_objective: str
    decisive_resource: str
    strongest_opponent: str
    hidden_dependency: str
    highest_value_unknown: str
    options: list[dict]
    selected_action: str
    selected_risk_class: str
    utility: float
    why_now: str
    retreat_trigger: str
    canon_anchor: str = ""


def compile_decision(state: State, options: list[Option], *,
                     decisive_resource: str = "primeval essence / capital",
                     strongest_opponent: str = "unspecified",
                     hidden_dependency: str = "unmapped",
                     canon_anchor: str = "") -> Decision:
    """Score all options under the context-adjusted utility function and pick
    per the policy: never auto-safe, never auto-aggressive."""
    lam = state.lambda_context()
    for o in options:
        o.risk_class = classify_risk(o, state)
    scored = sorted(options, key=lambda o: o.utility(lam, state.time_pressure), reverse=True)
    best = scored[0]

    # TERMINAL veto: reject terminal risk unless the path is already doomed
    if best.risk_class == "TERMINAL" and not state.path_doomed_without_change:
        survivors = [o for o in scored if o.risk_class != "TERMINAL"]
        best = survivors[0] if survivors else best

    why = _why_now(state, best, lam)
    trigger = _retreat_trigger(state, best)
    return Decision(
        true_objective=state.objective,
        decisive_resource=decisive_resource,
        strongest_opponent=strongest_opponent,
        hidden_dependency=hidden_dependency,
        highest_value_unknown=(state.unknowns[0] if state.unknowns else "none identified"),
        options=[asdict(o) | {"utility": round(o.utility(lam, state.time_pressure), 2)} for o in scored],
        selected_action=best.action,
        selected_risk_class=best.risk_class,
        utility=round(best.utility(lam), 2),
        why_now=why,
        retreat_trigger=trigger,
        canon_anchor=canon_anchor,
    )


def _why_now(state: State, opt: Option, lam: float) -> str:
    if state.path_doomed_without_change and opt.risk_class in ("TRANSFORMATIVE", "TERMINAL"):
        return ("the current path leads to certain death; a transformative risk is "
                "the only remaining continuation")
    if not state.secure_retreat and state.enemies_strength_relative > 1.2:
        return ("no secure retreat and stronger enemies present: survival-weighted "
                "selection; ruin exposure dominates the ledger")
    if state.secure_retreat and opt.risk_class in ("CALCULATED", "TRANSFORMATIVE"):
        return "a secure retreat exists: the calculated risk is affordable now"
    if opt.posture == "retreat":
        return "continuation threatens survival or future options; preserve the path"
    if opt.posture == "investigate":
        return "the highest-value unknown is cheap to resolve and gates every other option"
    return "expected continuation value dominates alternatives at current exposure"


def _retreat_trigger(state: State, opt: Option) -> str:
    if opt.posture in ("retreat", "investigate"):
        return "re-assess each turn; escalate if the unknown resolves favorably"
    deps = state.dependencies or []
    dep = f"dependency on {deps[0]} exceeds bound" if deps else "any dependency exceeds bound"
    return (f"retreat if {dep}, if survival probability drops, or if the core "
            f"assumption is invalidated by new information")


def to_json(d: Decision) -> str:
    return json.dumps(asdict(d), ensure_ascii=False, indent=2)


# ----------------------------------------------------------- formatter (B)
STYLE_EXAMPLES = {
    "brief": '"The price is acceptable. I will take it."',
    "mask_aware": "outer speech is chosen per audience; full reasoning never leaves the aperture",
}


def format_outer_speech(decision: Decision, audience: str = "neutral") -> str:
    """Two-mode policy: outer speech is brief and strategically incomplete."""
    lines = [f"{decision.selected_action}."]
    if decision.selected_risk_class in ("CALCULATED", "TRANSFORMATIVE"):
        lines.append("The risk is survivable. The window is not.")
    lines.append(f"If {decision.retreat_trigger.split(',')[0]}, I withdraw.")
    if audience in ("enemy", "suspicious_official"):
        lines = lines[:1]  # reveal the minimum to adversaries
    return " ".join(lines)
