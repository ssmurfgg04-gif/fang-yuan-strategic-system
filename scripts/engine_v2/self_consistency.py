"""Self-consistency minimax decision loop (engine v2).

Tree-of-Thoughts-style pipeline powered by GLM (z-ai SDK, via bun bridge):

  Stage 1 GENERATE    : 3 candidate strategies under 3 distinct personas
                        (shadow compounder / transactional arbitrager /
                        calculated aggressor).
  Stage 2 ADVERSARIAL : one attack per candidate — the model plays
                        opponent/environment and returns the strongest
                        failure mode, worst case, probability, survival flag.
  Stage 3 REFINE      : ONE budgeted revision per candidate to patch its
                        identified failure mode (skipped when the adversarial
                        survival-threat probability is negligible, < 0.05).
  Stage 4 MINIMAX     : transparent worst-case scoring + TERMINAL-style veto
                        (mirrors policy.decision_compiler semantics), argmax
                        of worst-case utility.

Scoring (transparent, deterministic):
    U = 4*resource + 3*info + 3*optionality + 2*growth
        - 5*exposure - 10*dependency
  All six factors are mapped from the candidate's JSON fields by a keyword
  lexicon (see `quantify_candidate`) onto [0, 1].  The adversarial worst case
  degrades factors with damage terms scaled by the attack probability:
    p_eff = probability * (1 - mitigation)          (mitigation from refine)
    positives shrink by p_eff * damage, negatives grow by p_eff * damage.

Veto semantics (mirrors the compiler's TERMINAL veto):
  * A candidate whose (survival_threatened AND p_eff >= 0.05) is hard-vetoed.
  * Unless EVERY candidate is doomed — then pick the least-doomed (lowest
    effective threat probability, tie-break on worst-case U), mirroring the
    compiler's doomed-path acceptance.

LLM access: bun + z-ai-web-dev-sdk via `llm_bridge.mjs` (stdin/stdout JSON).
  - concurrency <= 5 (ThreadPoolExecutor)
  - timeout 90 s per call, 3 retries
  - disk cache data/cache/selfconsistency/<sha1>.json
  - hard budget: <= 10 LLM calls per decision (3 gen + 3 attack + <=3 refine
    + 1 external-voice format).

This is fiction-strategy research (Reverend Insanity + Chinese classics):
outputs stay analytic/game-theoretic.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "cache" / "selfconsistency"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "engine_v2"

LLM_TIMEOUT_S = 90
LLM_RETRIES = 3
LLM_BACKOFF_S = (5.0, 20.0)      # gentle on the shared rate limit
MAX_WORKERS = 3                  # hard cap 5 per spec; 3 avoids 429 bursts
MAX_CALLS_PER_DECISION = 10
NEGLECTABLE_THREAT_P = 0.05      # survival threat below this is negligible
CACHE_VERSION = "sc_v2_1"

CANDIDATE_FIELDS = ("name", "action", "resource_move", "info_gain",
                    "concealment", "dependency_created", "exit_plan",
                    "time_horizon")
ATTACK_FIELDS = ("failure_mode", "worst_case", "probability",
                 "survival_threatened")


class LLMError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# LLM access layer: bun bridge + retries + disk cache
# --------------------------------------------------------------------------
def _bridge_command() -> list[str]:
    bun = shutil.which("bun") or "bun"
    bridge = Path(__file__).resolve().parent / "llm_bridge.mjs"
    return [bun, str(bridge)]


def _bridge_env() -> dict:
    """Environment for the bridge subprocess.

    z-ai-web-dev-sdk is installed in bun's global store; bare-import
    resolution falls back to it ONLY when no node_modules directory exists
    between the script and the filesystem root (any stray node_modules —
    e.g. /home/<u>/node_modules — cuts the fallback off).  Point NODE_PATH
    at the global store so the bridge works from any repo location."""
    env = os.environ.copy()
    candidates = []
    if env.get("NODE_PATH"):
        candidates.append(env["NODE_PATH"])
    bun_home = Path(env.get("BUN_INSTALL") or (Path.home() / ".bun"))
    candidates.append(str(bun_home / "install" / "global" / "node_modules"))
    candidates.append("/usr/local/lib/node_modules")
    dirs = [c for c in candidates if c and Path(c).is_dir()]
    if dirs:
        env["NODE_PATH"] = ":".join(dict.fromkeys(dirs))
    return env


def llm_complete(payload: dict) -> str:
    """Raw (uncached) completion. payload = {system, user, thinking}.

    One subprocess attempt per retry, 90 s timeout each, 3 retries.
    Returns the message content string; raises LLMError when exhausted.
    (Tests monkeypatch this function — it is the module-level seam.)
    """
    last_err = "unknown"
    for attempt in range(LLM_RETRIES):
        try:
            proc = subprocess.run(
                _bridge_command(),
                input=json.dumps(payload, ensure_ascii=False),
                capture_output=True, text=True, timeout=LLM_TIMEOUT_S,
                env=_bridge_env())
            out = json.loads(proc.stdout.strip())
            if out.get("ok"):
                return str(out.get("content", ""))
            last_err = str(out.get("error", "bridge returned ok=false"))
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
        if attempt < LLM_RETRIES - 1:
            time.sleep(LLM_BACKOFF_S[min(attempt, len(LLM_BACKOFF_S) - 1)])
    raise LLMError(last_err)


def cache_key(payload: dict) -> str:
    basis = {"v": CACHE_VERSION, "system": payload.get("system", ""),
             "user": payload.get("user", ""),
             "thinking": payload.get("thinking", "disabled")}
    return hashlib.sha1(json.dumps(basis, sort_keys=True,
                                   ensure_ascii=False).encode("utf-8")).hexdigest()


def cache_get(key: str, cache_dir: Path) -> str | None:
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    try:
        return str(json.loads(path.read_text(encoding="utf-8"))["content"])
    except Exception:  # noqa: BLE001
        return None


def cache_put(key: str, content: str, cache_dir: Path, meta: dict | None = None) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    path.write_text(json.dumps({"key": key, "content": content,
                                "created_at": time.time(),
                                "meta": meta or {}}, ensure_ascii=False),
                    encoding="utf-8")
    return path


def cached_llm_complete(payload: dict, cache_dir: Path = CACHE_DIR,
                        use_cache: bool = True) -> str:
    """Cache-wrapping completion; `llm_complete` stays the injectable seam."""
    key = cache_key(payload)
    if use_cache:
        hit = cache_get(key, cache_dir)
        if hit is not None:
            return hit
    content = llm_complete(payload)
    if use_cache:
        cache_put(key, content, cache_dir)
    return content


# --------------------------------------------------------------------------
# Strict JSON handling
# --------------------------------------------------------------------------
def extract_json(text: str) -> dict | None:
    """Pull the first balanced top-level {...} object out of an LLM reply."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        return obj if isinstance(obj, dict) else None
                    except Exception:  # noqa: BLE001
                        break
        start = text.find("{", start + 1)
    return None


def _as_text(v: Any) -> str | None:
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def validate_candidate(obj: Any) -> dict | None:
    """Strict schema: all 8 fields present as non-empty strings.

    Extra keys are tolerated but dropped (audit hygiene)."""
    if not isinstance(obj, dict):
        return None
    out = {}
    for f in CANDIDATE_FIELDS:
        v = _as_text(obj.get(f))
        if v is None:
            return None
        out[f] = v
    return out


def validate_attack(obj: Any) -> dict | None:
    if not isinstance(obj, dict):
        return None
    fm, wc = _as_text(obj.get("failure_mode")), _as_text(obj.get("worst_case"))
    if fm is None or wc is None:
        return None
    try:
        p = float(obj.get("probability"))
    except (TypeError, ValueError):
        return None
    st = obj.get("survival_threatened")
    if isinstance(st, str):
        st = st.strip().lower() in ("true", "yes", "1")
    if not isinstance(st, bool):
        return None
    atk = {"failure_mode": fm, "worst_case": wc,
           "probability": min(1.0, max(0.0, p)), "survival_threatened": st}
    dm = obj.get("damages")
    if isinstance(dm, dict):
        parsed = {}
        for k in ("resource", "info", "optionality", "growth",
                  "exposure", "dependency"):
            try:
                parsed[k] = min(1.0, max(0.0, float(dm.get(k, 0.0))))
            except (TypeError, ValueError):
                return None
        atk["damages"] = parsed
    return atk


def validate_refine(obj: Any) -> dict | None:
    cand = validate_candidate(obj)
    if cand is None:
        return None
    mit = obj.get("mitigation", 0.5)
    try:
        mit = min(1.0, max(0.0, float(mit)))
    except (TypeError, ValueError):
        mit = 0.5
    cand["mitigation"] = mit
    pfm = _as_text(obj.get("patched_failure_mode"))
    cand["patched_failure_mode"] = pfm or ""
    return cand


# --------------------------------------------------------------------------
# Transparent scoring: lexicon quantification of candidate fields -> [0,1]
# --------------------------------------------------------------------------
def _hits(text: str, words: tuple[str, ...]) -> int:
    t = text.lower()
    return sum(1 for w in words if w in t)


_RESOURCE_UP = ("secure", "guarantee", "accumulate", "compound", "harvest",
                "stockpile", "acquire", "extract", "monopolize", "monopoly",
                "income", "profit", "revenue", "rent", "tribute", "diversif",
                "reserves", "supply of")
_RESOURCE_DOWN = ("sacrifice", "spend", "pay out", "burn", "liquidate",
                  "deplete", "expensive", "heavy investment", "sell off",
                  "give up", "concede")
_INFO_UP = ("verify", "map", "scout", "observe", "learn", "intelligence",
            "probe", "audit", "review", "measure", "confirm", "identify",
            "investigate", "catalog", "test the", "determine")
_INFO_NONE = ("no new information", "none", "nothing learned")
_OPT_UP = ("exit", "withdraw", "fallback", "retreat", "alternative",
           "second source", "reversible", "staged", "pilot", "contingency",
           "preserve", "keep open", "parallel", "option", "revocable",
           "diversified")
_OPT_DOWN = ("exclusive", "lock", "lockout", "lock-in", "sole", "only supplier",
             "single point", "irreversible", "permanent", "all-in",
             "commit fully", "no exit", "bind", "captive", "oblige")
_GROWTH_UP = ("compound", "compounding", "growth", "grow", "scale", "snowball",
              "dividend", "recurring", "annuity", "cumulative", "multiplier",
              "flywheel", "momentum", "hidden advantage", "edge over time")
_CONCEAL_DOWN = ("conceal", "hide", "mask", "veil", "quiet", "discreet",
                 "secret", "unseen", "low profile", "subtle", "deniable",
                 "anonymous", "covert", "hidden", "unannounced", "private")
_CONCEAL_UP = ("announce", "public", "openly", "reveal", "display", "showcase",
               "declare", "flaunt", "conspicuous", "overt", "broadcast",
               "no concealment", "visibility", "registered")
_DEPEND_UP = ("exclusive", "sole", "single", "only one", "only supplier",
              "rely", "depend", "owe", "obligat", "captive", "lock-in",
              "lock in", "bound to", "hostage", "controlled by", "monopol")
_DEPEND_MILD = ("alliance", "partnership", "contract", "deal with",
                "cooperation", "trade ties")
_DEPEND_NONE = ("none", "no dependency", "no new dependency", "independent",
                "self-sufficient", "diversified", "multiple suppliers",
                "no single")


def quantify_candidate(cand: dict) -> dict:
    """Deterministic field -> factor mapping (all results clamped to [0,1]).

    resource / info / optionality / growth are 'goods';
    exposure / dependency are 'bads'."""
    act, rm = cand.get("action", ""), cand.get("resource_move", "")
    ig, co = cand.get("info_gain", ""), cand.get("concealment", "")
    dep, ex = cand.get("dependency_created", ""), cand.get("exit_plan", "")
    th = cand.get("time_horizon", "")

    resource = 0.40 + 0.13 * _hits(rm + " " + act, _RESOURCE_UP) \
        - 0.18 * _hits(rm + " " + act, _RESOURCE_DOWN)
    info = 0.35 + 0.20 * _hits(ig + " " + act, _INFO_UP) \
        - (0.25 if _hits(ig, _INFO_NONE) else 0.0)
    opt_text = " ".join([ex, act, dep])
    optionality = 0.35 + 0.17 * _hits(opt_text, _OPT_UP) \
        - 0.25 * _hits(opt_text, _OPT_DOWN)
    growth = 0.30 + 0.17 * _hits(" ".join([rm, act, th]), _GROWTH_UP)
    if "long" in th.lower():
        growth += 0.20
    if "short" in th.lower() or "immediate" in th.lower():
        growth -= 0.10
    exposure = 0.20 + 0.22 * _hits(co + " " + act, _CONCEAL_UP) \
        - 0.13 * _hits(co, _CONCEAL_DOWN)
    dependency = 0.05 + 0.28 * _hits(dep, _DEPEND_UP) \
        + 0.08 * _hits(dep, _DEPEND_MILD) \
        - (0.30 if _hits(dep, _DEPEND_NONE) else 0.0)
    clamp = lambda v: round(min(1.0, max(0.0, v)), 4)  # noqa: E731
    return {"resource": clamp(resource), "info": clamp(info),
            "optionality": clamp(optionality), "growth": clamp(growth),
            "exposure": clamp(exposure), "dependency": clamp(dependency)}


def base_utility(factors: dict) -> float:
    """U = 4*resource + 3*info + 3*optionality + 2*growth
           - 5*exposure - 10*dependency."""
    return round(4 * factors["resource"] + 3 * factors["info"]
                 + 3 * factors["optionality"] + 2 * factors["growth"]
                 - 5 * factors["exposure"] - 10 * factors["dependency"], 4)


# --------------------------------------------------------------------------
# Adversarial damage model (transparent, deterministic)
# --------------------------------------------------------------------------
_SEVERITY_STRONG = ("death", "die", "dies", "destroy", "ruin", "fatal",
                    "irrecoverable", "annihilat", "bankrupt", "extinct",
                    "killed", "ends the path", "collapse")
_SEVERITY_MODERATE = ("loss", "lose", "lock", "trap", "expos", "discover",
                      "uncover", "reveal", "seized", "confiscat", "default",
                      "breach", "betray", "sanction", "blocked", "cut off",
                      "shut out", "hostile takeover", "devour")
_SEVERITY_MILD = ("delay", "setback", "suspicion", "scrutiny", "watched",
                  "slower", "minor", "limited", "cost", "noise")

_DIM_TARGETS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("expos", "discover", "reveal", "uncover", "seen through", "found out",
      "identif", "suspicion", "scrutiny", "surveill", "watch", "public",
      "known to"), ("exposure",)),
    (("lock", "trap", "captiv", "sole", "single point", "dependent",
      "dependency", "monopol", "no exit", "cannot leave", "bound",
      "hostage", "leverage over"), ("dependency", "optionality")),
    (("resource", "capital", "asset", "supply", "essence", "money", "income",
      "revenue", "cash", "seized", "confiscat", "stockpile", "harvest",
      "profit", "income stream"), ("resource",)),
    (("blind", "ignoran", "deceiv", "mislead", "wrong model", "miscalibrat",
      "false picture", "unknow"), ("info",)),
    (("stall", "stagnat", "plateau", "stuck", "no growth", "wasted",
      "years behind"), ("growth",)),
    (("death", "die", "destroy", "ruin", "kill", "fatal", "annihilat",
      "extinct", "bankrupt", "collapse"), ("resource", "info", "optionality",
                                           "growth")),
)


def estimate_damages(attack: dict) -> dict:
    """Per-dimension worst-case damages in [0,1].

    If the attack supplied explicit numeric `damages`, use them (clamped);
    otherwise derive from a severity lexicon over failure_mode + worst_case:
    targeted dimensions take the full severity, untargeted positive dims take
    20 % collateral.  A survival threat implies everything is at stake
    (all positive dims >= 0.8 * severity)."""
    dmg = {k: 0.0 for k in ("resource", "info", "optionality", "growth",
                            "exposure", "dependency")}
    if isinstance(attack.get("damages"), dict):
        dmg.update(attack["damages"])
        return dmg
    text = (attack.get("failure_mode", "") + " " +
            attack.get("worst_case", "")).lower()
    if any(w in text for w in _SEVERITY_STRONG):
        sev = 1.0
    elif any(w in text for w in _SEVERITY_MODERATE):
        sev = 0.6
    elif any(w in text for w in _SEVERITY_MILD):
        sev = 0.3
    else:
        sev = 0.5
    targeted: set[str] = set()
    for words, dims in _DIM_TARGETS:
        if any(w in text for w in words):
            targeted.update(dims)
    for d in targeted:
        dmg[d] = sev
    for d in ("resource", "info", "optionality", "growth"):
        if d not in targeted:
            dmg[d] = max(dmg[d], 0.2 * sev)
    if attack.get("survival_threatened"):
        floor = 0.8 * sev
        for d in ("resource", "info", "optionality", "growth", "exposure",
                  "dependency"):
            dmg[d] = max(dmg[d], floor if d in targeted else 0.4 * sev)
    return dmg


def worst_case_utility(factors: dict, attack: dict,
                       mitigation: float = 0.0) -> tuple[float, float]:
    """Return (u_worst, p_eff).

    p_eff = attack.probability * (1 - mitigation); positive factors shrink by
    p_eff * damage, negative factors grow by p_eff * damage; all clamped."""
    p = min(1.0, max(0.0, float(attack.get("probability", 0.0)) *
                     (1.0 - min(1.0, max(0.0, mitigation)))))
    d = estimate_damages(attack)
    clamp = lambda v: min(1.0, max(0.0, v))  # noqa: E731
    r, i = clamp(factors["resource"] - p * d["resource"]), \
        clamp(factors["info"] - p * d["info"])
    o, g = clamp(factors["optionality"] - p * d["optionality"]), \
        clamp(factors["growth"] - p * d["growth"])
    x, dep = clamp(factors["exposure"] + p * d["exposure"]), \
        clamp(factors["dependency"] + p * d["dependency"])
    u = round(4 * r + 3 * i + 3 * o + 2 * g - 5 * x - 10 * dep, 4)
    return u, round(p, 4)


# --------------------------------------------------------------------------
# Stage 4: minimax selection with TERMINAL-style veto
# --------------------------------------------------------------------------
def is_doomed(entry: dict) -> bool:
    atk, p_eff = entry["attack"], float(entry.get("p_eff", 1.0))
    return bool(atk.get("survival_threatened")) and p_eff >= NEGLECTABLE_THREAT_P


def minimax_select(entries: list[dict]) -> tuple[int | None, str]:
    """entries: [{gen, attack, refined?, u_pre, u_worst, p_eff, doomed?}].

    Returns (selected index or None, selection note)."""
    if not entries:
        return None, "no candidates produced"
    doomed_flags = [is_doomed(e) for e in entries]
    viable = [i for i, d in enumerate(doomed_flags) if not d]
    if viable:
        best = max(viable, key=lambda i: (entries[i]["u_worst"],
                                          entries[i]["u_pre"]))
        return best, ("minimax argmax of worst-case utility among "
                      f"{len(viable)}/{len(entries)} non-vetoed candidates")
    least = min(range(len(entries)),
                key=lambda i: (entries[i]["p_eff"], -entries[i]["u_worst"]))
    return least, ("ALL candidates survival-threatened: least-doomed fallback "
                   "(lowest effective threat probability), mirroring the "
                   "compiler's doomed-path TERMINAL acceptance")


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
PERSONAS = (
    ("shadow_compounder",
     "Persona: SHADOW COMPOUNDER. Conceal real strength; build quiet, hidden, "
     "compounding advantage. Prefer low-visibility accumulation, optionality, "
     "and patience. Never reveal the true objective; let rivals underestimate."),
    ("transactional_arbitrager",
     "Persona: TRANSACTIONAL ARBITRAGER. Treat trade, leverage, alliances and "
     "reputation as instruments with prices. Extract value from exchanges, "
     "keep counterparties replaceable, avoid single-point dependency."),
    ("calculated_aggressor",
     "Persona: CALCULATED AGGRESSOR. Strike decisively only when the edge is "
     "real and the window is open; every strike is prepared with a rehearsed "
     "exit route and bounded downside. Aggression is a tool, not a mood."),
)

GEN_SYSTEM = (
    "You are the strategy-generation engine of a fiction-strategy research "
    "project (a game-theoretic decision engine inspired by the novel Reverend "
    "Insanity and Chinese strategic classics: Sunzi, Han Feizi, Guiguzi). "
    "You evaluate high-stakes strategic trade-offs under hidden information. "
    "Stay analytic and game-theoretic. Output STRICT JSON only — no prose, "
    "no code fences. All values are short strings (<= 20 words each).")

GEN_USER_TMPL = (
    "Strategic scenario: {scenario}\n\n"
    "{persona}\n\n"
    "Produce the single best candidate strategy for this persona as STRICT "
    "JSON with EXACTLY these keys:\n"
    '{{"name": "...", "action": "...", "resource_move": "...", '
    '"info_gain": "...", "concealment": "...", "dependency_created": "...", '
    '"exit_plan": "...", "time_horizon": "..."}}\n'
    "Guidance: action = the concrete move now; resource_move = how resources "
    "are gained/spent/positioned; info_gain = what becomes known/verified; "
    "concealment = what is hidden from rivals; dependency_created = any new "
    "single-point dependence (write 'none' if none); exit_plan = the way out "
    "if it fails; time_horizon = short/medium/long.")

ATK_SYSTEM = (
    "You are the red-team adversary engine of a fiction-strategy research "
    "project (game-theoretic, inspired by Reverend Insanity). You play the "
    "opponent and the hostile environment: your job is to break the proposed "
    "strategy. Be concrete and ruthless but analytic. Output STRICT JSON "
    "only — no prose, no code fences.")

ATK_USER_TMPL = (
    "Scenario: {scenario}\n"
    "Candidate strategy under attack: {cand}\n\n"
    "You are the opponent/environment. Find the strongest failure mode and "
    "the worst-case outcome. Reply as STRICT JSON with EXACTLY these keys:\n"
    '{{"failure_mode": "...", "worst_case": "...", "probability": <float '
    '0-1>, "survival_threatened": <true|false>}}\n'
    "probability = chance the failure mode actually materializes; "
    "survival_threatened = true only if the worst case destroys the agent's "
    "continuity/operational capability, not merely profits.")

REFINE_SYSTEM = (
    "You are the strategy-refinement engine of a fiction-strategy research "
    "project (game-theoretic, inspired by Reverend Insanity). Patch the "
    "identified failure mode while preserving the strategy's core intent. "
    "Output STRICT JSON only — no prose, no code fences.")

REFINE_USER_TMPL = (
    "Scenario: {scenario}\n"
    "Original strategy: {cand}\n"
    "Adversarial finding: {attack}\n\n"
    "Revise the strategy to patch this failure mode (one revision only: "
    "cheap patches — hedges, verification steps, staged commitment, better "
    "exit). Reply as STRICT JSON with the SAME eight keys as the original "
    '("name", "action", "resource_move", "info_gain", "concealment", '
    '"dependency_created", "exit_plan", "time_horizon") plus two more:\n'
    '{{..., "patched_failure_mode": "<how the failure mode is now handled>", '
    '"mitigation": <float 0-1, fraction of the threat removed>}}')

FMT_SYSTEM = (
    "You are the external-voice formatter of a fiction-strategy decision "
    "engine. Two-mode policy: the external message is brief, plausible and "
    "strategically incomplete — it never reveals probabilities, worst cases, "
    "or the full reasoning. Output STRICT JSON only.")

FMT_USER_TMPL = (
    "Scenario: {scenario}\n"
    "Selected strategy: {sel}\n"
    "Strongest surviving worst case: {wc}\n\n"
    'Reply as STRICT JSON: {{"rationale_2sentences": "<exactly two '
    'analytic sentences: why this survives the minimax comparison>", '
    '"external_message": "<short veiled public message, <= 25 words, no '
    'internal details>"}}')


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------
FALLBACK_CANDIDATES = {
    "shadow_compounder": {
        "name": "quiet accumulation (fallback)",
        "action": "investigate quietly while accumulating resources at low visibility",
        "resource_move": "diversify holdings; no committed spend",
        "info_gain": "map the counterparty's true position before committing",
        "concealment": "reveal nothing; routine business posture",
        "dependency_created": "none",
        "exit_plan": "hold reversible positions; stop-loss at first sign of exposure",
        "time_horizon": "long"},
    "transactional_arbitrager": {
        "name": "hedged exchange (fallback)",
        "action": "negotiate a small reversible pilot trade with verification",
        "resource_move": "trade only a bounded slice; keep reserves liquid",
        "info_gain": "price the counterparty's reliability from the pilot",
        "concealment": "frame as routine commerce; hide valuation interest",
        "dependency_created": "none; counterparties kept replaceable",
        "exit_plan": "pilot ends cleanly; switch suppliers if terms sour",
        "time_horizon": "medium"},
    "calculated_aggressor": {
        "name": "prepared decisive move (fallback)",
        "action": "prepare a decisive move but trigger only with verified edge",
        "resource_move": "stage resources without committing them",
        "info_gain": "confirm the window with independent checks first",
        "concealment": "preparations disguised as routine operations",
        "dependency_created": "none",
        "exit_plan": "rehearsed withdrawal trigger; abandon if the edge fades",
        "time_horizon": "short"},
}

FALLBACK_ATTACK = {"failure_mode": "unassessed (adversary call unavailable)",
                   "worst_case": "unknown; assume moderate damage",
                   "probability": 0.4, "survival_threatened": False}


class SelfConsistencyLoop:
    """GLM-powered generate -> attack -> refine -> minimax decision loop."""

    def __init__(self, *, cache_dir: Path | None = None, use_cache: bool = True,
                 max_workers: int = MAX_WORKERS,
                 max_calls: int = MAX_CALLS_PER_DECISION,
                 complete_fn: Callable[[dict], str] | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.use_cache = use_cache
        self.max_workers = max(1, min(max_workers, MAX_WORKERS))
        self.max_calls = max_calls
        self._complete_fn = complete_fn  # injection hook (tests may also
        # monkeypatch the module-level llm_complete directly)
        self._lock = threading.Lock()
        self.total_llm_calls = 0

    # ------------------------------------------------------------ plumbing
    def _one(self, system: str, user: str, thinking: str = "enabled") -> str | None:
        """Single budgeted call; returns None on failure or budget exhaustion."""
        with self._lock:
            if self.total_llm_calls >= self.max_calls:
                return None
            self.total_llm_calls += 1
        payload = {"system": system, "user": user, "thinking": thinking}
        try:
            fn = self._complete_fn or llm_complete
            if self.use_cache and self._complete_fn is None:
                return cached_llm_complete(payload, self.cache_dir, True)
            return fn(payload)
        except Exception:  # noqa: BLE001
            return None

    def _map(self, jobs: list[tuple[str, str, str]]) -> list[str | None]:
        """Parallel map over (system, user, thinking) with bounded workers."""
        if len(jobs) <= 1:
            return [self._one(*j) for j in jobs]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(jobs))) as ex:
            return list(ex.map(lambda j: self._one(*j), jobs))

    # -------------------------------------------------------------- stages
    def _generate(self, scenario: str) -> list[dict]:
        jobs = [(GEN_SYSTEM, GEN_USER_TMPL.format(scenario=scenario,
                                                  persona=text), "enabled")
                for _, text in PERSONAS]
        raws = self._map(jobs)
        cands = []
        for (pers, _), raw in zip(PERSONAS, raws):
            cand = validate_candidate(extract_json(raw or ""))
            if cand is None:
                fb = dict(FALLBACK_CANDIDATES[pers])
                fb["persona"] = pers
                fb["fallback"] = True
                cands.append(fb)
            else:
                cand["persona"] = pers
                cand["fallback"] = False
                cands.append(cand)
        return cands

    def _attack(self, scenario: str, cands: list[dict]) -> list[dict]:
        jobs = [(ATK_SYSTEM,
                 ATK_USER_TMPL.format(scenario=scenario,
                                      cand=json.dumps(c, ensure_ascii=False)),
                 "enabled") for c in cands]
        raws = self._map(jobs)
        attacks = []
        for raw in raws:
            atk = validate_attack(extract_json(raw or ""))
            if atk is None:
                atk = dict(FALLBACK_ATTACK)
                atk["fallback"] = True
            else:
                atk["fallback"] = False
            attacks.append(atk)
        return attacks

    def _refine(self, scenario: str, cand: dict, atk: dict) -> dict | None:
        raw = self._one(REFINE_SYSTEM,
                        REFINE_USER_TMPL.format(
                            scenario=scenario,
                            cand=json.dumps(cand, ensure_ascii=False),
                            attack=json.dumps(atk, ensure_ascii=False)),
                        "enabled")
        return validate_refine(extract_json(raw or "")) if raw else None

    # ---------------------------------------------------------------- main
    def run(self, scenario: str) -> dict:
        self.total_llm_calls = 0
        scenario = (scenario or "").strip() or "(empty scenario)"

        # Stage 1 — GENERATE
        cands = self._generate(scenario)

        # Stage 2 — ADVERSARIAL
        attacks = self._attack(scenario, cands)

        # Stage 3 + 4 assembly
        entries: list[dict] = []
        for cand, atk in zip(cands, attacks):
            factors = quantify_candidate(cand)
            u_pre = base_utility(factors)
            entry = {
                "gen": cand, "attack": atk,
                "factors": factors, "u_pre": u_pre,
                "refined": None,
                "p_eff": round(atk["probability"], 4),
                "u_worst_raw": worst_case_utility(factors, atk)[0],
            }
            # Stage 3 — REFINE (budgeted): skip when the survival-threat
            # probability is negligible (< 0.05).
            needs_refine = not (atk["survival_threatened"]
                                and atk["probability"] < NEGLECTABLE_THREAT_P)
            if needs_refine:
                ref = self._refine(scenario, cand, atk)
                if ref is not None:
                    entry["refined"] = ref
                    entry["p_eff"] = round(
                        atk["probability"] * (1.0 - ref["mitigation"]), 4)
            # effective factors/u for selection: refined candidate overrides
            eff_factors = quantify_candidate(entry["refined"]) \
                if entry["refined"] else factors
            entry["effective_factors"] = eff_factors
            u_worst, _ = worst_case_utility(
                eff_factors, atk,
                mitigation=entry["refined"]["mitigation"]
                if entry["refined"] else 0.0)
            entry["u_worst"] = u_worst
            entry["doomed"] = is_doomed(entry)
            entries.append(entry)

        sel, note = minimax_select(entries)
        selected = entries[sel] if sel is not None else None

        # External voice + rationale (final budgeted call, deterministic
        # fallback if unavailable)
        rationale, external = self._format(scenario, selected, note)

        audit = {
            "engine": "self_consistency_minimax_v2",
            "scenario": scenario,
            "candidates": [{
                "gen": e["gen"],
                "attack": e["attack"],
                "refined": e["refined"],
                "factors": e["factors"],
                "effective_factors": e["effective_factors"],
                "u_pre": e["u_pre"],
                "u_worst_raw": e["u_worst_raw"],
                "u_worst": e["u_worst"],
                "p_eff": e["p_eff"],
                "doomed": e["doomed"],
            } for e in entries],
            "selected": sel,
            "selection_note": note,
            "rationale_2sentences": rationale,
            "external_message": external,
            "total_llm_calls": self.total_llm_calls,
        }
        return audit

    # ------------------------------------------------------- external voice
    def _format(self, scenario: str, selected: dict | None,
                note: str) -> tuple[str, str]:
        if selected is None:
            return ("No candidate was produced; the audit is retained for "
                    "inspection."), "Position unchanged. We watch, we wait."
        raw = self._one(FMT_SYSTEM,
                        FMT_USER_TMPL.format(
                            scenario=scenario,
                            sel=json.dumps(selected["gen"], ensure_ascii=False),
                            wc=selected["attack"].get("worst_case", "unknown")),
                        "disabled")
        obj = extract_json(raw or "") if raw else None
        if obj and _as_text(obj.get("rationale_2sentences")) \
                and _as_text(obj.get("external_message")):
            return (_as_text(obj["rationale_2sentences"]),
                    truncate_words(_as_text(obj["external_message"]), 25))
        # deterministic fallback (two sentences, veiled)
        action = selected["gen"]["action"].rstrip(".")
        exit_plan = selected["gen"].get("exit_plan", "prepared exit").rstrip(".")
        worst = selected["attack"].get("worst_case", "uncertainty").rstrip(".")
        rationale = (f"Selected '{selected['gen'].get('name', 'candidate')}' "
                     f"as the argmax of worst-case utility ({selected['u_worst']}), "
                     f"surviving the strongest attack: {worst}. "
                     f"The move stays reversible with the exit '{exit_plan}', "
                     f"so the surviving downside is bounded and affordable.")
        external = truncate_words(
            f"We proceed as proposed. Terms are acceptable for now; if "
            f"conditions shift, we step aside. {action}.", 25)
        return rationale, external


def truncate_words(text: str, limit: int) -> str:
    words = text.split()
    return " ".join(words[:limit])


# --------------------------------------------------------------------------
# CLI: python -m scripts.engine_v2.self_consistency "scenario" [--out PATH]
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="self-consistency minimax loop")
    ap.add_argument("scenario")
    ap.add_argument("--out", default=None, help="output audit JSON path")
    args = ap.parse_args(argv)
    loop = SelfConsistencyLoop()
    audit = loop.run(args.scenario)
    out = Path(args.out) if args.out else \
        DEFAULT_OUT_DIR / f"smoke_minimax_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps({"selected": audit["selected"],
                      "selection_note": audit["selection_note"],
                      "external_message": audit["external_message"],
                      "total_llm_calls": audit["total_llm_calls"],
                      "out": str(out)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
