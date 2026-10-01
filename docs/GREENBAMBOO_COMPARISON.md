# Surpassing GreenBamboo-v1 — Comparison & Borrowed Lessons

GreenBamboo-v1 (u/Awkward_Cancel8495, r/ReverendInsanity) is the most visible
prior attempt at a Reverend Insanity LLM: a **12B Gemma-3 finetune** released as
GGUF, designed as a **roleplay GM**. This document records what we studied, what
we borrowed, and where this project's architecture goes beyond it.

## Side by side

| Dimension | GreenBamboo-v1 | This system |
|---|---|---|
| Core bet | finetune carries the world (12B weights) | model carries only policy (0.5B); DB/simulator/search carry the world |
| Training data | what-if scenario narratives + lore text | 211 structured decision records (5 classes) + chapter-anchored corpus + classics principles + negatives |
| Canon recall | explicitly weak ("not a plot database" — author) | FTS5 retrieval over the full 4.28M-word corpus with chapter citations + verification discipline |
| Hallucination control | none documented | quote verification statuses; "Chinese wording not verified" rule; provenance table |
| Evaluation | none public | 140-item fidelity benchmark, 10 secret holdout, 7-dimension scoring, persona baselines (policy 0.651 vs safe 0.329 vs reckless 0.219) |
| Behavior measurement | vibes ("stylistic direction is on point") | outcome metrics: survival, resources, drawdown, options, ruin; 34 automated tests |
| Counterfactuals | in training data (their key idea) | in training data AND benchmark (graded variant families) AND simulator (dynamic mutation) |
| Adaptation under change | untested | reversal tests, counterfactual layer, 5-environment simulator with hidden state |
| Deliverable | chat model for roleplay | policy actor + retrieval + simulator + benchmark + training pipeline (the full loop) |
| Legal posture | community finetune on lore text | provenance-gated: CC BY-SA + public domain ingested; user corpus stays local (user_responsibility) |

## What we borrowed (with credit)

1. **What-if scenario datasets are the core teaching instrument.** GreenBamboo's
   best idea. Our benchmark's counterfactual families and the new `what-if`
   item layer (10 items) are direct descendants — but ours are graded, so the
   same items generate SFT/DPO pairs automatically (`build_datasets.py`).
2. **Tone/ideology is learnable at meaningful scale.** Their success confirms
   finetuning can reshape the "world-flavor" of a model — our Stage-A adds a
   GM-mode formatter target so both products can ship from one base.
3. **GGUF distribution** (Q8_0 / Q5_K_M) as the practical delivery format for
   community use — adopted in our runbook's export step.

## Where we surpass

1. **Policy over persona.** GreenBamboo trains "how to perform RI." We train
   "which action preserves the path." The benchmark's counterfactual layer
   exists precisely because tonal fidelity can hide policy failure: a model can
   speak perfectly demonic and still pick the assistant-safe option every time.
2. **Canon grounding without weight bloat.** The author's admitted weakness
   (no plot recall) and the community's first request (add RAG) are solved at
   the architecture level: a 0.5B policy actor + 4.28M-word retrievable corpus
   with per-chapter citations beats asking 12B weights to memorize a banned
   10k-page novel.
3. **Hallucination discipline.** Quote records carry verification status; the
   system is trained to say "the Chinese wording is not verified in the current
   corpus" — a rule GreenBamboo's thread shows users actually need.
4. **Measurement.** Every claim we make about the system is a number from the
   benchmark or the simulator, reproducible from the repo, with persona
   baselines proving the metrics discriminate.
5. **The full loop.** Generate → simulate → score → retain winners → build
   preference pairs → fine-tune → holdout-test. GreenBamboo is one finetune;
   this is the training loop the loopless approach converges to.
6. **Failure as first-class data.** 35 failure records + `fake_fang_yuan.jsonl`
   negatives teach the model what NOT to repeat — absent from lore-only
   finetunes, which accidentally teach sunk-cost attachment and theatrical
   cruelty by example.

## Risks we inherit from their thread (mitigations in place)

| GreenBamboo risk | Our mitigation |
|---|---|
| Wrong chat template silently degrades output | template validation in eval harness; adapter ships exact template |
| Small finetunes can't follow length instructions | policy actor capped at 80–180 token JSON; length is the formatter's job |
| Stacked creative finetunes lobotomize | Stage-A starts from pristine Qwen2.5-0.5B-Instruct, never community merges |
| "Does not understand text-length commands" | GBNF grammar-constrained JSON output (llama.cpp) removes drift |
| No eval → regressions invisible | benchmark + 34 tests + DB-logged runs make regressions loud |

## Bottom line

GreenBamboo-v1 answered "can a finetuned model *feel* like Reverend Insanity?"
— yes. This project answers the harder question: **does it choose what Fang
Yuan would choose — and can you prove it?** The two are complements: a GM-mode
formatter on top of our policy actor would serve GreenBamboo's audience too,
while our benchmark finally makes fidelity claims falsifiable.
