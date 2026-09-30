# FINDINGS — Research & Engineering Log

Session date: 2026-10-01. Deep-research-first directive executed (17 search
queries + 18 wiki page fetches), then the system was built and tested.

## 1. Research findings (web)

### Corpus & canon
- **Novel status**: Reverend Insanity (蛊真人, author Gu Zhen Ren) was banned
  and cancelled in 2019 at **chapter 2334**, unfinished (novelupdates + multiple
  sources). The corpus and benchmark must respect the real ending: no invented
  final arcs.
- **Fandom wiki**: reverend-insanity.fandom.com is CC BY-SA 3.0 and fetchable
  via the permitted MediaWiki API format (`api.php`), but direct scraping from
  this sandbox hits Cloudflare. The server-side reader (`page_reader`) fetched
  all 18 target pages cleanly (5.4k–16.3k words each; ~150k words total).
  All ingested with license attribution in `sources`.
- **Chinese classics**: all 8 target classics are public domain. zh.wikisource
  hosts 道德經 (匯校版, split into 81 章), 孫子兵法 (13 篇), 韓非子 (subpages
  五蠹/說難/孤憤/二柄/定法), 鬼谷子, 商君書 (卷一/卷二), 史記·貨殖列傳
  (卷129), 戰國策 (士禮居叢書本). 資治通鑑 vol.1 was NOT found under the
  standard title on zh.wikisource (known gap; the 四部叢刊本 scans exist).
- **Legal landscape** (Bartz v. Anthropic / Kadrey v. Meta): training-copy
  fair use looks favorable, but pirated-source retention remains contested.
  Engineering response implemented: provenance-gated ingestion with
  quarantine state; the pipeline never fetches copyrighted material itself;
  user-provided files are user responsibility after explicit confirmation.

### Engineering (small-model policy training)
- Qwen2.5-0.5B + LoRA (r16, lr 2e-4, 3 epochs) is the community-standard
  recipe; Unsloth/axolotl/peft+trl all viable. Compact JSON outputs via GBNF
  grammar in llama.cpp are the key to usable CPU inference (15–35 tok/s on
  2 cores → 3–5 s per decision).
- DPO/ORPO preference tuning is the standard follow-up for behavior surgery;
  GRPO-style simulator RL is documented but requires the GPU training burst.
- RAG evaluation practice: separate retrieval relevance, faithfulness, and
  answer correctness — mirrored in the benchmark's layer design.

## 2. Engineering findings (built & tested this session)

### F-1. Decision-compiler λ dynamics work as specified
Context-weighted ruin penalty (λ=8 surrounded / 2 secure retreat / 0.5 doomed)
plus a **delay penalty** under time pressure reproduce canon behavior:
- Qing Mao-style trap with battle pressure → escape selected (canon);
- same state without pressure → waiting is affordable;
- TERMINAL veto unless the path is already doomed;
- transformative risk accepted only when the path is doomed and continuation
  is layered.

### F-2. Option-estimation conditionality is the #1 failure mode (critical)
The policy agent initially died 100% of the time in the alliance environment
because `set_contingency` was priced with future-options=4 although
contingency alone does NOT satisfy the survival requirement (escape_ready or
survival resource). Similarly, the career agent took calculated risks without
the foundation because the option was always on the menu.
**Rule established**: *the compiler cannot see conditions it is never shown —
option parameters must encode the real mechanism, including tails
(ruin_probability of using the survival resource early = 0.45)*.
This is exactly the training-data lesson for the 0.5B: labels must carry
mechanism, not vibes.

### F-3. Simulator discriminates archetypes (20 seeds × 5 envs × 4 agents)
| env | random | safe_generic | reckless | fang_yuan_policy_v1 |
|---|---|---|---|---|
| gu_world resources | 20.0 | 13.0 | 13.0 | **51.2** |
| industrial resources | 1.4 (45% dead) | 40.0 | 60.0 | **72.0** |
| career options | 1.95 | 2.0 | 1.0 (75% dead) | **3.0** |
| alliance_betrayal | 28.0 (15% dead) | 20.0 | 20.0 | **40.0, alive** |
| arbitrage | 92.4 | 84.0 | 100.0 | 90.0 (no ruin) |

The policy agent is never always-safe (unlike safe_generic) and never always-
aggressive (unlike reckless, which dies 75% in career): calibrated aggression.

### F-4. Benchmark separates systems (110 items, 10 secret holdout)
| adapter | F | holdout |
|---|---|---|
| fang_yuan_policy_v1 | **0.625** | **0.944** |
| safe_generic | 0.325 | 0.590 |
| reckless_theatrical | 0.217 | 0.443 |

Concealment discipline: policy 1.00, safe 0.76, reckless 0.36.
Risk calibration (counterfactuals): policy 0.95, safe 0.50, reckless 0.52.
Caveat (integrity): the mock policy agent answers counterfactual posture
questions from the policy spec (reference oracle) — its scores calibrate the
grading pipeline; LLM-backed adapters are the real measurement targets.

### F-5. Retrieval bugs that would have poisoned the model
1. **Contentless FTS5 cannot return columns** — rowid-only retrieval with
   join-back to base tables is mandatory.
2. **FTS5 syntax errors were silently swallowed** — natural-language queries
   with `?`/quotes broke MATCH and returned empty. Fix: query sanitizer
   (strip punctuation, drop stopwords, OR-join, CJK phrase quoting).
3. **Rowid desync** — inserts/deletes through several code paths desynced
   FTS rowids from base ids (152 vs 140) and a contentless-table 'delete'
   with wrong values **corrupted the index** ("database disk image is
   malformed"). Fix: one rebuild script (`reindex_fts.py`) that always
   inserts with explicit rowid = base id, and db_utils now does the same on
   every insert. Deleting via the FTS 'delete' command is banned.
4. **bm25 ranking matters**: OR-queries without `ORDER BY rank` return
   common-word matches first; Liquor Worm ranked behind note chunks.

### F-6. Broken Formation demo (5 seeds incl. the canonical 77)
fang_yuan_policy_v1: survival 1.00, capital 104.0, options 3.00, drawdown
0.120, never accepts exclusivity. safe: 40.0. reckless: 60.0 but accepts
exclusivity (leash). random: 36.1. The investigation-first-then-concentrate
pattern dominates.

## 3. Test coverage
34 pytest tests pass (policy λ dynamics, terminal veto, time pressure, risk
buckets, outer-speech concealment, simulator determinism, hidden-state leak
checks, ruin reachability, FTS alignment/relevance, quote-integrity
discipline, classics-never-canon, provenance validity, benchmark layer
completeness, holdout existence, policy-beats-baselines, training-data
shapes, local-corpus quarantine gate). All logged in `test_runs`.

## 4. Known gaps / next steps
1. **Scale-up**: 114 seed decisions → 500/1500/300/200/100 targets require
   the chapter-extraction pass over the user-provided corpus (provenance gate
   is ready; `ingest_local_corpus.py`).
2. **資治通鑑 vol.1** not found on zh.wikisource; fetch from another PD source.
3. **GPU fine-tune**: Stages A–C are prepared (datasets + runbook) but not
   executed here (no GPU); benchmark will re-score via LlamaCppAdapter.
4. **Human eval**: the 3-group blind evaluation (RI readers / Chinese-lit
   readers / strategy evaluators) is designed in the notes but not recruited.
