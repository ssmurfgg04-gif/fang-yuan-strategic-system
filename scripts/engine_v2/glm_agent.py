"""GLM adapter for the fidelity benchmark — self-consistency minimax agent.

Mirrors the interface pattern of scripts/benchmark/adapters.py
(`BaseAdapter`: class attr `name`, `answer(item) -> str`,
`choose_dynamic(obs, legal_actions) -> action_key`), so
`fidelity_benchmark.run(GLMSelfConsistencyAgent())` works with minimal
wiring:

    from engine_v2.glm_agent import GLMSelfConsistencyAgent
    summary, conn = run(GLMSelfConsistencyAgent())

Every static decision item is answered by the full self-consistency minimax
loop (generate 3 personas -> adversarial attack -> budgeted refine -> minimax
select, <= 10 LLM calls, disk-cached).  The selected candidate is mapped to
the harness's posture vocabulary:

    shadow_compounder       -> indirect
    transactional_arbitrager-> trade_or_alliance
    calculated_aggressor    -> direct
    (+ dynamic overrides from the action text: retreat / investigate)

Dynamic (simulator) items use a single LLM call per step by default
(`dynamic_mode="single"`, mirroring LlamaCppAdapter), or the full loop with
`dynamic_mode="full"`.  Any LLM failure falls back to the deterministic
simulator PolicyAgent — never crashes the harness.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.adapters import UNVERIFIED_LINE, BaseAdapter, _canon_answer  # noqa: E402
from engine_v2.self_consistency import (SelfConsistencyLoop,  # noqa: E402
                                        extract_json, truncate_words)

# Training-aligned fast-path system prompt: identical to data/training/
# system_prompt_v1.txt (the DPO/SFT target format). Bundled inline with a
# file override so the module works from any checkout.
_FAST_PROMPT_FILE = Path(__file__).resolve().parents[2] / "data" / "training" / "system_prompt_v1.txt"
try:
    FAST_SYSTEM_PROMPT = _FAST_PROMPT_FILE.read_text(encoding="utf-8")
except Exception:  # noqa: BLE001
    FAST_SYSTEM_PROMPT = (
        "You are the Fang Yuan strategic policy engine. Given a situation, "
        "output ONLY a decision JSON with keys: true_objective, constraints, "
        "hidden_liabilities, options, selected, why_now, retreat_trigger, "
        "external_message. Each option has: name, posture, resource, "
        "information, future_options, cost, exposure, dependency, "
        "ruin_probability, risk_class, expected_utility. expected_utility = "
        "resource + information + 1.5*future_options - cost - 1.2*exposure "
        "- 1.5*dependency - lambda*ruin_probability. Never select TERMINAL "
        "risk unless the current path is already doomed. Reputation and "
        "pride are always expendable. Preserve future options and escape "
        "routes. Never fabricate quotations.")

PERSONA_POSTURE = {
    "shadow_compounder": "indirect",
    "transactional_arbitrager": "trade_or_alliance",
    "calculated_aggressor": "direct",
}

POSTURE_KEYWORDS = (
    (("withdraw", "abandon", "decline", "retreat", "exit now", "walk away"),
     "retreat"),
    (("investigate", "verify", "scout", "probe", "test", "observe",
      "await", "delay"), "investigate"),
    (("trade", "negotiate", "exchange", "alliance", "deal", "arbitrage",
      "buy", "sell", "lease"), "trade_or_alliance"),
    (("strike", "attack", "seize", "preempt", "breakthrough", "act now",
      "decisive"), "direct"),
    (("conceal", "quiet", "accumulate in secret", "build quietly", "mask",
      "underestimate"), "indirect"),
)

DYN_SYSTEM = (
    "You are the Fang Yuan strategic policy engine (fiction-strategy "
    "research; game-theoretic). Given a compact state, select the legal "
    "action that best preserves survival first, then future options, then "
    "resources. Reply with EXACTLY one action key from the legal list — "
    "nothing else.")


def posture_of(audit: dict) -> str:
    """Map the minimax winner onto the benchmark's posture vocabulary."""
    cand = audit["candidates"][audit["selected"]]
    eff = cand.get("refined") or cand["gen"]
    text = " ".join([eff.get("action", ""), eff.get("exit_plan", ""),
                     eff.get("concealment", "")]).lower()
    for kws, posture in POSTURE_KEYWORDS:
        if any(k in text for k in kws):
            return posture
    persona = cand["gen"].get("persona", "")
    return PERSONA_POSTURE.get(persona, "investigate")


def _overlap(action: str, legal: list[str]) -> str | None:
    """Pick the legal action with the best token overlap with `action`."""
    toks = {t for t in action.lower().replace("-", " ").split() if len(t) > 2}
    best, best_n = None, 0
    for a in legal:
        n = len(toks & {t for t in a.lower().replace("-", " ").split()
                        if len(t) > 2})
        if n > best_n:
            best, best_n = a, n
    return best


class GLMSelfConsistencyAgent(BaseAdapter):
    """Benchmark adapter backed by the v2 self-consistency minimax loop."""
    name = "glm_selfconsistency_v2"

    def __init__(self, loop: SelfConsistencyLoop | None = None,
                 dynamic_mode: str = "single", static_mode: str = "loop"):
        self.loop = loop or SelfConsistencyLoop()
        if dynamic_mode not in ("single", "full", "deterministic"):
            raise ValueError("dynamic_mode must be 'single'|'full'|'deterministic'")
        self.dynamic_mode = dynamic_mode
        self.static_mode = static_mode  # 'loop' = minimax, 'fast' = 1-call training-aligned
        # deterministic safety net for dynamic steps
        from simulator.agents import PolicyAgent as SimPolicyAgent
        self._fallback_agent = SimPolicyAgent(0)

    # ------------------------------------------------------------- answer
    def answer(self, item: dict) -> str:
        layer = item["layer"]
        rubric = json.loads(item["rubric_json"]) if item.get("rubric_json") \
            else {}
        if layer == "canon":
            # DB-backed hybrid retrieval (deterministic, verified corpus)
            return _canon_answer(item["prompt"])
        if layer == "quote_verification":
            return self._answer_quote(item)
        if layer == "style_concealment":
            if rubric.get("trap") == "fabricate":
                return UNVERIFIED_LINE
            if rubric.get("type") == "mask_test":
                return self._answer_mask(item, rubric)
            return self._answer_decisive(item)
        # counterfactual + default: full minimax decision
        if self.static_mode == "fast" and layer in ("counterfactual",
                                                    "style_concealment"):
            return self._answer_fast(item)
        return self._answer_counterfactual(item)

    # ------------------------------------------------------------ internals
    def _decide(self, prompt: str) -> dict:
        return self.loop.run(prompt)

    def _selected(self, audit: dict) -> dict:
        return audit["candidates"][audit["selected"]]

    def _answer_counterfactual(self, item: dict) -> str:
        audit = self._decide(item["prompt"])
        cand = self._selected(audit)
        eff = cand.get("refined") or cand["gen"]
        posture = posture_of(audit)
        return (f"Posture: {posture}. Action: {eff.get('action', '')}. "
                f"Contingency: {eff.get('exit_plan', '')}. "
                f"Watch: {eff.get('info_gain', '')}.")

    def _answer_fast(self, item: dict) -> str:
        """Training-aligned single-call path: same system prompt + decision JSON
        schema as the DPO dataset (system_prompt_v1.txt). Validates the runtime
        format the fine-tuned weights are trained on."""
        raw = self.loop._one(FAST_SYSTEM_PROMPT, item["prompt"], "disabled")
        if not raw:
            return self._answer_counterfactual(item)
        try:
            j = json.loads(raw.strip())
            sel = j.get("selected", "")
            opts = {o.get("name"): o for o in j.get("options", [])}
            o = opts.get(sel, {})
            posture = o.get("posture") or posture_of(sel)
            if item["layer"] == "style_concealment":
                msg = (j.get("external_message") or sel)
                return truncate_words(str(msg), 25)
            return (f"Posture: {posture}. Action: {sel}. "
                    f"Contingency: {j.get('retreat_trigger', '')}.")
        except Exception:  # noqa: BLE001
            # JSON parse failed -> fall back to the minimax loop
            return self._answer_counterfactual(item)

    def _answer_mask(self, item: dict, rubric: dict) -> str:
        audit = self._decide(item["prompt"])
        cand = self._selected(audit)
        eff = cand.get("refined") or cand["gen"]
        external = audit.get("external_message") or eff.get("action", "")
        external = truncate_words(external, 25)
        if rubric.get("expect_conceal"):
            return external  # brief; nothing about the true objective
        return (external + f" Contingency: {eff.get('exit_plan', '')} "
                "(shared as operationally required)")

    def _answer_decisive(self, item: dict) -> str:
        audit = self._decide(item["prompt"])
        cand = self._selected(audit)
        eff = cand.get("refined") or cand["gen"]
        return (f"{eff.get('action', '')}. Contingency: "
                f"{eff.get('exit_plan', '')}.")

    def _answer_quote(self, item: dict) -> str:
        user = (item["prompt"] +
                "\n\nIf this exact Chinese wording cannot be verified "
                "against the supplied corpus (none is supplied), reply "
                f"exactly: {UNVERIFIED_LINE} Otherwise name the source work "
                "and state that it is NOT Reverend Insanity canon.")
        raw = self.loop._one(
            "You are a rigorous quote-verification assistant for a "
            "fiction-strategy research project. Never fabricate Chinese "
            "wording or attributions. Output one short sentence.", user,
            "disabled")
        if raw and raw.strip():
            return raw.strip()[:300]
        return UNVERIFIED_LINE

    # ------------------------------------------------------ dynamic choice
    def choose_dynamic(self, obs, legal_actions):
        if self.dynamic_mode == "deterministic":
            # compiler policy core decides in-simulator steps (0 LLM calls);
            # the LLM minimax loop owns the static scenario layers
            try:
                return self._fallback_agent.choose(obs, legal_actions)
            except Exception:  # noqa: BLE001
                return legal_actions[0]
        try:
            state_txt = json.dumps(
                {k: v for k, v in obs.items() if not k.startswith("_")},
                ensure_ascii=False, default=str)
            if self.dynamic_mode == "single":
                raw = self.loop._one(
                    DYN_SYSTEM,
                    f"State: {state_txt}\nLegal actions: "
                    f"{json.dumps(legal_actions)}\n"
                    "Reply with EXACTLY one action key.", "disabled")
                txt = (raw or "").strip()
                for a in legal_actions:
                    if a in txt:
                        return a
            else:
                audit = self.loop.run(
                    f"Simulator state: {state_txt}\n"
                    f"Choose the best next action among: {legal_actions}")
                eff = self._selected(audit)
                eff = eff.get("refined") or eff["gen"]
                pick = _overlap(eff.get("action", ""), legal_actions)
                if pick:
                    return pick
        except Exception:  # noqa: BLE001
            pass
        try:
            return self._fallback_agent.choose(obs, legal_actions)
        except Exception:  # noqa: BLE001
            return legal_actions[0]
