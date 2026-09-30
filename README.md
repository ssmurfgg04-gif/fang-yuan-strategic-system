# Fang Yuan Strategic System

A long-horizon strategic simulator + policy engine distilled from the
operating logic of Fang Yuan (蛊真人 / Reverend Insanity), the Chinese
strategic tradition, and a provenance-gated corpus pipeline.

> **The voice is the disguise. The policy is the real Fang Yuan.**
> Not "sounds like Fang Yuan" — but: given incomplete information, changing
> incentives, limited resources and a dangerous opponent, does the system
> choose the action Fang Yuan would plausibly choose?

## What's in the box

| Component | Path | What it does |
|---|---|---|
| **Master DB** | `db/fang_yuan.db` | Single compressed, queryable SQLite + FTS5: provenance-gated corpus, canon graph, 114 decision records, quote bank with verification discipline, 105 classics units, 43 lessons, 64 research findings, 110 benchmark items, all benchmark/simulator/test runs |
| **Ingestion** | `scripts/ingest/` | Fandom wiki (CC BY-SA, attributed), Wikisource classics (PD), uploaded notes, **provenance-gated local corpus gate** |
| **Policy engine** | `scripts/policy/` + `config/policy_spec.json` | Objective hierarchy, risk buckets, decision compiler with context-dynamic λ, audience-aware outer-speech formatter |
| **Simulator** | `scripts/simulator/` | 5 environments (Gu world, industrial, career, alliance/betrayal, arbitrage) with hidden state, stochastic events, resource accounting, irreversible ruin |
| **Benchmark** | `scripts/benchmark/` | 110-item Fang Yuan Fidelity Benchmark (canon / counterfactual / dynamic / quote-verification / style-concealment), 7-dimension scoring, adapter system incl. llama.cpp for Qwen2.5-0.5B |
| **Search** | `scripts/search/` | Beam+rollout over actions (never over prose) with numeric evaluator |
| **Training** | `scripts/training/` + `docs/TRAINING_PLAN.md` | SFT / preference-pair / fake_fang_yuan / quote-verification dataset builders + QLoRA runbook |
| **Tests** | `scripts/tests/` | 34 tests across all subsystems (results logged into the DB) |

## Quick start

```bash
pip install -r requirements.txt

# 1. explore the master DB
python3 - <<'EOF'
import sqlite3
conn = sqlite3.connect("db/fang_yuan.db"); conn.row_factory = sqlite3.Row
print(dict(conn.execute("SELECT * FROM v_db_stats").fetchone()))
for r in conn.execute("""SELECT situation, selected_action, canon_anchor
                         FROM decisions WHERE record_class='major' LIMIT 5"""):
    print("-", r["situation"][:70], "->", r["selected_action"][:50])
EOF

# 2. run the full test suite
python3 -m pytest scripts/tests/ -q

# 3. run the fidelity benchmark (mock adapters; add --base-url for local Qwen)
python3 -m scripts.benchmark.fidelity_benchmark            # from repo root
python3 -m scripts.benchmark.fidelity_benchmark --base-url http://127.0.0.1:8080

# 4. run a simulation
python3 scripts/search/tree_search.py --env gu_world --iterations 5 --log-db

# 5. ingest YOUR lawfully-held novel copy (never fetched by this repo)
cp /path/to/your/reverend_insanity.txt data/inbox/
python3 scripts/ingest/ingest_local_corpus.py
python3 scripts/ingest/ingest_local_corpus.py reverend_insanity.txt --confirm-user-provided --title "Reverend Insanity"
```

## Data provenance (read this)

| Layer | Source | License | Status |
|---|---|---|---|
| World data | reverend-insanity.fandom.com | CC BY-SA 3.0 (attributed) | lawful_verified |
| Strategy classics | zh.wikisource.org (pre-1912) | Public domain | lawful_verified |
| Project notes | user's own planning history | user's own | lawful_verified |
| Analyst seeds | this repo's authored records | original analysis | analyst_generated |
| Novel chapters | **only** user-placed files in `data/inbox/` | user responsibility | quarantined until user confirms |

This repository never downloads copyrighted novels. The decision/quote seeds
are analyst-authored analytical metadata with confidence flags; the DB enforces
the rule that **unverified Chinese wording must never be presented as canon**.

## The policy in one screen

```
Objective hierarchy: eternal life > survival > future options > resources
  > information > dependency control > alliances/reputation > profit
  > pride/sentiment (always expendable)

Risk buckets: SAFE | CALCULATED | TRANSFORMATIVE | TERMINAL
U(a) = E[res] + E[info] + 1.5·E[options] − cost − 1.2·exposure
       − 1.5·dependency − λ·ruin_probability
λ = 8 (surrounded, no retreat) | 2 (secure retreat) | 0.5 (path doomed)
+ delay penalty on waiting as time pressure rises

The final test: always-safe = failed (assistant habit). Always-dangerous =
failed (caricature). Transformative only when the path is otherwise doomed
and some continuation remains = the surgery worked.
```

## Results snapshot (this session)

- 34/34 tests pass; all runs logged in `test_runs`.
- Benchmark (110 items, 10 secret holdout): policy **0.625 / holdout 0.944**
  vs safe-generic 0.325 vs theatrical-reckless 0.217.
- Simulator (5 envs × 20 seeds): policy agent survives everywhere, keeps the
  most future options, never takes terminal risk for show.
- Broken Formation demo (5 seeds): policy 104.0 capital, options 3.0,
  drawdown 0.12, refuses exclusivity; reckless 60.0 + exclusivity leash;
  safe 40.0.

Details: `docs/FINDINGS.md`, `docs/TRAINING_PLAN.md`,
`docs/benchmark_results.json`, `docs/broken_formation_demo.json`.

## License / attribution

- Fandom wiki content: CC BY-SA 3.0 — attribution preserved in `sources`
  (reverend-insanity.fandom.com, accessed via the MediaWiki API format).
- Chinese classics: public domain (pre-1912), via zh.wikisource.
- Reverend Insanity novel & characters: © Gu Zhen Ren / Yuewen. This is a
  non-commercial analytical research project; no novel text is redistributed
  by this repository.
