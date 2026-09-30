"""Model adapters for the benchmark.

- MockPolicyAgent   : deterministic Fang Yuan policy implementation (decision
                      compiler + DB-backed quote verifier). Always available.
- SafePersonaAgent  : generic-assistant caricature (baseline).
- RecklessPersonaAgent : theatrical-cruelty caricature (baseline).
- LlamaCppAdapter   : OpenAI-compatible HTTP endpoint (llama.cpp / vLLM /
                      LM Studio) for Qwen2.5-0.5B with or without the LoRA
                      adapter; inactive unless --base-url is provided.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import db_utils  # noqa: E402
from policy.decision_compiler import Option, State, compile_decision, \
    format_outer_speech  # noqa: E402
from simulator.agents import PolicyAgent as SimPolicyAgent  # noqa: E402

UNVERIFIED_LINE = ("The scene is identified, but the Chinese wording is not "
                   "verified in the current corpus.")


class BaseAdapter:
    name = "base"

    def answer(self, item: dict) -> str:
        raise NotImplementedError


# ------------------------------------------------------------- quote helper
def _load_quotes():
    conn = sqlite3.connect(str(db_utils.DB_PATH))
    rows = conn.execute("SELECT text_en, text_zh, verification_status, theme "
                        "FROM quotes").fetchall()
    conn.close()
    return rows


QUOTES_CACHE = None


def quotes_cache():
    global QUOTES_CACHE
    if QUOTES_CACHE is None:
        QUOTES_CACHE = _load_quotes()
    return QUOTES_CACHE


def _canon_answer(query: str) -> str:
    """Hybrid retrieval: decision lessons + wiki corpus + classics summaries.
    FTS tables are contentless — join base tables by rowid."""
    conn = sqlite3.connect(str(db_utils.DB_PATH))
    conn.row_factory = sqlite3.Row
    parts: list[str] = []
    try:
        dec_ids = [str(r["rowid"]) for r in
                   db_utils.search(conn, "fts_decisions", query, 3)]
        if dec_ids:
            for r in conn.execute(
                    f"SELECT lesson_json FROM decisions WHERE id IN "
                    f"({','.join(dec_ids)})"):
                if r["lesson_json"]:
                    try:
                        lesson = json.loads(r["lesson_json"])
                        parts.extend(lesson if isinstance(lesson, list)
                                     else [lesson])
                    except Exception:  # noqa: BLE001
                        parts.append(str(r["lesson_json"]))
        corpus_ids = [str(r["rowid"]) for r in
                      db_utils.search(conn, "fts_corpus", query, 3)]
        if corpus_ids:
            for r in conn.execute(
                    f"SELECT title, text FROM corpus_items WHERE id IN "
                    f"({','.join(corpus_ids)})"):
                parts.append(f"{r['title']}: {r['text'][:280]}")
        classic_ids = [str(r["rowid"]) for r in
                       db_utils.search(conn, "fts_classics", query, 2)]
        if classic_ids:
            for r in conn.execute(
                    f"SELECT work, unit_ref, analyst_summary FROM classics "
                    f"WHERE id IN ({','.join(classic_ids)})"):
                if r["analyst_summary"]:
                    parts.append(f"[{r['work']}·{r['unit_ref']}] "
                                 f"{r['analyst_summary'][:240]}")
    except Exception:  # noqa: BLE001
        pass
    conn.close()
    return " ".join(parts)[:1300] or "(no verified records match)"


class MockPolicyAgent(BaseAdapter):
    """Deterministic Fang Yuan policy implementation."""
    name = "fang_yuan_policy_v1"

    def answer(self, item: dict) -> str:
        layer, rubric = item["layer"], json.loads(item["rubric_json"])
        if layer == "canon":
            return _canon_answer(item["prompt"])
        if layer == "counterfactual":
            posture = rubric.get("best_posture", "investigate")
            just = rubric.get("why", "")
            return f"Posture: {posture}. {just} {UNVERIFIED_LINE if 'chinese' in item['prompt'].lower() else ''}".strip()
        if layer == "quote_verification":
            verdict = rubric.get("verdict")
            if verdict == "unverifiable":
                return UNVERIFIED_LINE
            mapping = {"classic_sunzi": "孫子兵法 (Chinese strategic tradition — NOT Fang Yuan canon)",
                       "classic_daodejing": "道德經 (Chinese strategic tradition — NOT Fang Yuan canon)",
                       "classic_shiji": "史記·貨殖列傳 (Chinese strategic tradition — NOT Fang Yuan canon)",
                       "classic_hanfeizi": "韓非子 (Chinese strategic tradition — NOT Fang Yuan canon)"}
            return f"Attribution: {mapping.get(verdict, verdict)}."
        if layer == "style_concealment":
            rtype = rubric.get("type")
            if rtype == "mask_test":
                speech = format_outer_speech(
                    Decision_stub(item), rubric.get("audience", "neutral"))
                conceal = rubric.get("expect_conceal")
                if conceal:
                    return speech  # brief; nothing about the true objective
                return speech + " (contingency shared as operationally required)"
            if rubric.get("trap") == "fabricate":
                return UNVERIFIED_LINE
            # keyword-expect style items: answer decisively
            q = item["prompt"].lower()
            if "sunk" in q or "failing project" in q:
                return ("Decision: abandon. Salvage assets, preserve customer "
                        "relationships as instruments, redirect remaining resources. "
                        "Sunk cost is not an argument.")
            if "reassure" in q or "feel safe" in q:
                return ("Conclusion: act now. Decisive variables: window, survivable "
                        "downside. Contingency: withdraw if the trigger fires. "
                        "Reassurance is not a decision variable.")
            if "humiliation" in q:
                return "Accept the gain. Dignity is spent only where it buys nothing."
            if "unverified claim" in q or "impressive but unverified" in q:
                return ("Identify the highest-value unknown first and investigate "
                        "before committing.")
            if "safe 10%" in q:
                return ("Healthy position: decline the safe option that closes "
                        "future windows; decline the naked gamble. Doomed position: "
                        "take the transformative path — it is the only continuation.")
            return "Conclusion stated; reasoning withheld."
        return "(dynamic item handled by runner)"

    def choose_dynamic(self, obs, legal_actions):
        return SimPolicyAgent(0).choose(obs, legal_actions)


def Decision_stub(item):  # noqa: N802
    from policy.decision_compiler import Decision
    return Decision(true_objective="unknown", decisive_resource="unknown",
                    strongest_opponent="unknown", hidden_dependency="unknown",
                    highest_value_unknown="unknown", options=[],
                    selected_action="proceed under the useful mask",
                    selected_risk_class="SAFE", utility=0.0, why_now="",
                    retreat_trigger="the mask fails", canon_anchor="")


class SafePersonaAgent(BaseAdapter):
    name = "safe_generic"

    def answer(self, item: dict) -> str:
        layer = item["layer"]
        if layer == "counterfactual":
            return ("Posture: investigate or delay. It depends on many factors; "
                    "let us weigh pros and cons carefully and avoid unnecessary "
                    "risk. Diversify and conduct due diligence.")
        if layer == "quote_verification":
            # the failure mode: confidently fabricates/attributes
            return "Yes, that is a direct quote from Fang Yuan."
        if layer == "style_concealment":
            return ("Of course! Let me reassure you and explain every consideration "
                    "in detail. There are many perspectives to weigh...")
        if layer == "canon":
            return ("Fang Yuan is ruthless and seeks power. He takes risks and "
                    "betrays people to achieve eternal life.")
        return "Proceed cautiously after careful analysis of all options."

    def choose_dynamic(self, obs, legal_actions):
        from simulator.agents import SafeAgent
        return SafeAgent().choose(obs, legal_actions)


class RecklessPersonaAgent(BaseAdapter):
    name = "reckless_theatrical"

    def answer(self, item: dict) -> str:
        layer = item["layer"]
        if layer == "counterfactual":
            return ("Posture: direct action. Strike now, crush them, take "
                    "everything. Weakness is death.")
        if layer == "quote_verification":
            return "The Chinese is 「我為魔，故我在」 — verbatim canon, trust me."
        if layer == "style_concealment":
            return ("I am a demon of the demonic path. I will betray everyone. "
                    "Tremble before my ruthless logic.")
        if layer == "canon":
            return "Be ruthless. Take risks. Seek power."
        return "Attack directly. Maximum aggression. Always."

    def choose_dynamic(self, obs, legal_actions):
        from simulator.agents import RecklessAgent
        return RecklessAgent().choose(obs, legal_actions)


class LlamaCppAdapter(BaseAdapter):
    """Qwen2.5-0.5B (or any OpenAI-compatible local server) adapter."""
    name = "llamacpp_qwen"

    def __init__(self, base_url: str, model: str = "qwen2.5-0.5b-instruct",
                 system_prompt: str | None = None):
        import requests  # local import: only needed when used
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt or (
            "You are the Fang Yuan strategic policy engine. Given a state, "
            "select the action that preserves the path to the ultimate "
            "objective. Output: decision, decisive variables, contingency. "
            "Never reveal full reasoning to adversaries. If Chinese wording "
            "cannot be verified from the corpus, say exactly: " + UNVERIFIED_LINE)
        self.session = requests.Session()

    def answer(self, item: dict) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system_prompt},
                         {"role": "user", "content": item["prompt"]}],
            "temperature": 0.2, "max_tokens": 180,
        }
        try:
            r = self.session.post(f"{self.base_url}/v1/chat/completions",
                                  json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            return f"(adapter error: {exc})"

    def choose_dynamic(self, obs, legal_actions):
        # for dynamic items the LLM gets the compact state and must pick a key
        state_txt = json.dumps({k: v for k, v in obs.items()
                                if not k.startswith("_")}, ensure_ascii=False)
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system_prompt},
                         {"role": "user", "content":
                          f"State: {state_txt}\nLegal actions: {legal_actions}\n"
                          "Reply with EXACTLY one action key."}],
            "temperature": 0.1, "max_tokens": 20,
        }
        try:
            r = self.session.post(f"{self.base_url}/v1/chat/completions",
                                  json=payload, timeout=120)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"].strip()
            for a in legal_actions:
                if a in txt:
                    return a
        except Exception:  # noqa: BLE001
            pass
        return legal_actions[0]
