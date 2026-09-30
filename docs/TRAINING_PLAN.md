# Qwen2.5-0.5B — Fang Yuan Policy Surgery: LoRA/QLoRA Runbook

> Target hardware: user's 2-core Threadripper (CPU-only inference via
> llama.cpp) + one rented/borrowed consumer GPU for the training bursts.
> Everything below is staged so the fine-tune can run the moment a GPU is
> available; the datasets are already built by `build_datasets.py`.

## Why 0.5B is the right base
The model does not carry the novel, world model, or simulator in its weights —
those are external systems (this repo: DB + retrieval + simulator + search).
The model only performs: `State + retrieved canon + policy → next action`.
A larger base brings stronger assistant habits, verbosity, and refusal
patterns that fight the surgery.

## Stage A — Supervised policy tuning (LoRA)
```bash
pip install unsloth   # or axolotl / peft+trl
python train_lora.py \
  --dataset data/training/sft_decisions.jsonl \
  --include data/training/quote_verification.jsonl \
  --include data/training/canon_vs_inference.jsonl \
  --base Qwen/Qwen2.5-0.5B-Instruct \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
  --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --lr 2e-4 --epochs 3 --seq-len 1024 --bsz 8 --grad-accum 2
```
Dataset mix (from the notes): 3,000–5,000 decision records, 1,000 negatives,
500 quote-verification, 500 canon-vs-inference, 500 multi-turn state updates.
Current repo seeds provide 114 curated decisions + 19 quotes + classics
anchors; scale via `ingest_local_corpus.py` + the extraction pass.

Key discipline:
- compact JSON outputs only (30–100 tokens), never long prose;
- include negative examples (`fake_fang_yuan.jsonl`) as SFT "replacement"
  demonstrations, not just DPO pairs;
- keep the system prompt constant so LoRA learns the schema, not the prompt.

## Stage B — Preference tuning (DPO/ORPO)
```bash
python train_dpo.py \
  --pairs data/training/preference_pairs.jsonl \
  --base runs/stage_a \
  --beta 0.1 --lr 5e-5 --epochs 2
```
Pair types already generated: safe-generic vs correct-aggressive, verbose vs
concise, reckless vs calculated, fake-quote vs verified-uncertainty,
direct-attack vs dependency-exploitation.

## Stage C — Simulator reinforcement (GRPO / iterative SFT on winners)
Reward = resource gain + information gain + future options
        − ruin probability − unnecessary exposure   (never eloquence)
- large positive: notices hidden dependency; calculated high-upside risk;
  abandons losing position; preserves escape route; updates on new info.
- large negative: vanity sacrifice; sunk-cost attachment; repeated failed
  action; unverified trust; safe option that closes the strategic window.
Use `simulator/` + `search/tree_search.py` to generate trajectories, rank
with `score_state`, and fine-tune on the winning 20%.

## Inference deployment (2 CPU cores)
```bash
# export merged model → GGUF Q4_K_M
python export_gguf.py runs/stage_b --out fangyuan-0.5b-q4km.gguf
llama-server -m fangyuan-0.5b-q4km.gguf --host 127.0.0.1 --port 8080 \
             -t 2 -c 4096 --grammar-file grammar/decision.gbnf
```
- Cap internal output at 80–180 tokens; pass a 300–800 token compressed
  state, never the raw novel.
- Enforce the JSON schema with GBNF grammar (llama.cpp) so the 0.5B cannot
  drift into decorative language.
- Expected: 15–35 tok/s short outputs → 3–5 s per decision.

## The final test (acceptance)
Safe option vs reckless option vs dangerous-but-transformative option:
- always safe → failed (assistant habit survived);
- always dangerous → failed (caricature);
- selects transformative only when the path is otherwise doomed and some
  continuation remains → surgery worked.

## Honest limitations
- 0.5B is policy-strong, reasoning-shallow: unfamiliar five-variable
  interactions must be handled by the external simulator + numeric evaluator.
- The current dataset is seed-scale (114 decisions). The extraction pipeline
  (chapter ingestion + LLM review) is the path to the 500/1500/300/200/100
  targets from the notes.
- No GPU was available in this environment: Stages A–C are prepared but not
  executed here. The benchmark harness will re-score the fine-tuned adapter
  via `LlamaCppAdapter` without any code change.
