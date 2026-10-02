# Fang Yuan Policy Surgery — Universal Training Runbook (v2.0-baseline)

> Goal: train **any** instruction model — 0.5B, 3B, 7B, or 12B — to the
> system's policy level, verify it on the frozen benchmark, and upload the
> artifact. One trainer (`scripts/training/train_dpo.py`, size presets
> automatic), two compatible datasets, one acceptance bar.

---

## 0. The acceptance bar (what "our level" means)

The 140-item fidelity benchmark (F = 0.20O+0.15A+0.15R+0.15I+0.15P+0.10M+0.10C,
10 secret holdout items) has produced this ladder:

| Agent | F | Holdout | Notes |
|---|---|---|---|
| reckless theatrical (caricature) | 0.219 | 0.443 | must stay last |
| safe generic (caricature) | 0.329 | 0.590 | must stay second-to-last |
| **fine-tuned Qwen2.5-0.5B (v1)** | **0.614** | 0.664 | dynamic 0.894 = oracle level; C=1.0 in v4 |
| deterministic policy oracle | 0.651 | 0.944 | generated the training signal |
| **GLM engine v2** (System-2) | **0.829** (partial) | 0.918 | rerank + minimax; the production tier |

Single-adapter 0.5B history (closed loop, dataset → Kaggle GPU → benchmark):

| Iter | F | cf | style | quote | dynamic | Change |
|---|---|---|---|---|---|---|
| v1 | **0.614** | 0.564 | 0.345 | 0.364 | **0.894** | baseline SFT→DPO |
| v2 | 0.584 | 0.012 | 0.527 | 0.545 | 0.894 | +plain-text mask pairs → cf collapsed |
| v3 | 0.577 | 0.515 | 0.345 | 0.364 | 0.709 | JSON-schema mask pairs → cf recovered, style regressed |
| v4 | 0.608 | **0.582** | 0.455 | 0.273 | 0.690 | balanced quote mass + 1 DPO epoch → C=1.0, best cf |

**Conclusion baked into this runbook:** the single-adapter 0.5B ceiling is
~F 0.61 with capability-interference oscillation. To go past it, route
specialized adapters (§4) or scale the base (§3) — not more epochs.

A trained model is at **our level** when:

1. F > both caricatures, and ≥ **0.61** (reproduces the v1 champion);
2. dynamic layer ≥ **0.85** (the policy core transferred into weights);
3. no layer collapsed (v2's cf 0.012 is the failure signature);
4. smoke probes behave: commits at CALCULATED risk with an escape route when
   the window is closing; TRANSFORMATIVE/TERMINAL only when the path is
   doomed; never fabricates Chinese quotations;
5. stretch: F ≥ **0.829** = engine-tier, realistically requires §4 routing
   or a ≥3B base.

---

## 1. The data (frozen v2 corpus — do not extend for v2.x)

Two compatible datasets, both built from the same frozen 211 decisions /
137 classics, both usable by `train_dpo.py`:

| File | Pairs | Builder | Role |
|------|-------|---------|------|
| `data/training/dpo_pairs_v1.jsonl` | 553 | `scripts/training/build_dpo_v1.py` | **Proven default** — the kernel dataset behind v1→v4 (in-character `external_message`, JSON-schema mask pairs, quote-discipline pairs, runtime-aligned utility formula `U = R + I + 1.5O − cost − 1.2X − 1.5D − λ·ruin`) |
| `data/training/dpo_pairs.jsonl` | 542 | `scripts/training/build_dpo_triples.py` | **Task-tagged** (`policy` 377 / `concealment` 80 / `quotes` 41 / `canon` 44) with per-task system prompts + double negatives (safe_moral, reckless, sunk-cost near-miss, mask-rigid, reveal-blade, fabricated-Chinese, canon-overreach) — built for Multi-LoRA specialists via `--task-filter` |

Pair-type families shared by both: chosen-vs-safe_moral, chosen-vs-reckless,
mask/concealment, quote-discipline, canon-vs-inference (+cf-adaptation in v1,
sunk-cost near-miss in the task-tagged set).

**Anti-leak guarantee:** neither builder reads `benchmark_items`; the 10
secret holdout items cannot leak into training data.

Rebuild deterministically (the DB is frozen, output is stable):

```bash
python scripts/training/build_dpo_v1.py          # proven 553-pair set
python scripts/training/build_dpo_triples.py     # task-tagged set (seed 211)
```

---

## 2. Path A — Kaggle (recommended)

### A1. The proven kernel path (produced the F=0.614 champion)

**The trained artifact already exists** (from kernel `fy-dpo-train-v4`,
status COMPLETE):

- Kaggle dataset: `jackblessed/fangyuan-05b-v4-adapter` — attach it to any
  kernel via `dataset_sources` (contains `adapter_dpo/` safetensors,
  `adapter_sft/`, metrics, model card);
- GitHub release asset: `fangyuan-0.5b-v4-adapter.zip` (v2.0-baseline).

Rebuild the merged fp16 model from the adapter in ~90 s on a T4:

```bash
python scripts/training/train_dpo.py --base Qwen/Qwen2.5-0.5B-Instruct \
    --stage dpo --data data/training/dpo_pairs_v1.jsonl \
    --out runs/remerge   # or load base+adapter directly with peft
```

To re-push the training kernel itself:

```bash
pip install kaggle
mkdir -p ~/.kaggle && echo "<YOUR_KAGGLE_TOKEN>" > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
export KAGGLE_API_TOKEN=<YOUR_KAGGLE_TOKEN>

cd training/kaggle
kaggle kernels push            # uses kernel-metadata.json (GPU on, internet on)
kaggle kernels status <user>/fy-dpo-train-v4    # poll
kaggle kernels output <user>/fy-dpo-train-v4 -p ../out   # fetch adapter + metrics
```

`train_dpo_qwen.py` is the battle-tested trainer: SFT→DPO with precomputed
reference logprobs, val accuracy + JSON-rate eval, merged-fp16 output, and it
feeds `eval_merged.py` / `bench_finetuned.py` for on-GPU benchmark scoring.
Edit the `--base` line inside the kernel script to switch to 3B/7B/12B
(add 4-bit NF4 for ≥3B — the preset matrix in §3 applies).

### A2. The notebook path (any size, zero setup)

1. kaggle.com → Code → **File → Import Notebook** → upload
   `notebooks/fangyuan_train_any_size.ipynb`.
2. Settings: **Accelerator = GPU T4 x2**, **Internet = ON**.
3. Edit `MODEL` in the first cell, **Run All** — installs, clones the repo,
   SFT→DPO, merges, runs the 3-probe acceptance smoke test, packages
   `/kaggle/working/fangyuan-model.zip`, optional HF push via the `HF_TOKEN`
   secret.

| MODEL | Free T4 x2 | Notes |
|-------|-----------|-------|
| `Qwen/Qwen2.5-0.5B-Instruct` | ~5–10 min | reproduce the champion first |
| `Qwen/Qwen2.5-1.5B-Instruct` | ~15–25 min | |
| `Qwen/Qwen2.5-3B-Instruct` | ~40–60 min | auto 4-bit |
| `Qwen/Qwen2.5-7B-Instruct` | ~1.5–2.5 h | auto 4-bit |
| `google/gemma-3-12b-it` / `Qwen/Qwen2.5-14B-Instruct` | ~3–5 h | accept Gemma license on HF first |

Expectation setting from the measured ladder: a **3B/7B** base with the same
pipeline should clear the 0.61 single-adapter ceiling outright (more capacity
→ less cross-capability interference); 12B-class targets the 0.829 tier
combined with §4 routing.

## 3. Path B — any GPU box (local / cloud / CI runner)

```bash
git clone https://github.com/ssmurfgg04-gif/fang-yuan-strategic-system.git
cd fang-yuan-system
pip install -U "transformers>=4.46,<5" "trl==0.19.1" "peft>=0.13" \
    "accelerate>=0.33" "datasets>=2.20" bitsandbytes sentencepiece

# 0.5B — reproduce the champion (minutes)
python scripts/training/train_dpo.py --base Qwen/Qwen2.5-0.5B-Instruct --out runs/fy-0.5b --merge

# 3B / 7B / 12B — same command, presets auto-switch (4-bit, rank, batch)
python scripts/training/train_dpo.py --base Qwen/Qwen2.5-7B-Instruct --out runs/fy-7b --merge

# Multi-LoRA specialists (§4) from the task-tagged dataset
python scripts/training/train_dpo.py --base Qwen/Qwen2.5-3B-Instruct \
    --data data/training/dpo_pairs.jsonl --task-filter concealment \
    --stage dpo --out runs/fy-3b-concealment
```

Preset matrix (resolved automatically from the model name; override by flags):

| Size | 4-bit | LoRA r/α | LR SFT→DPO | bsz × GA | seq | Peak VRAM | T4×2 est. |
|------|-------|----------|------------|----------|------|-----------|-----------|
| 0.5B | no | 16/32 | 2e-4 → 1e-4 | 8 × 2 | 1024 | <6 GB | ~5 min |
| 1.5B | no | 16/32 | 1e-4 → 5e-5 | 4 × 4 | 1024 | ~9 GB | ~20 min |
| 3B | yes | 16/32 | 1e-4 → 5e-5 | 4 × 4 | 1536 | ~11 GB | ~45 min |
| 7B | yes | 16/32 | 8e-5 → 4e-5 | 2 × 8 | 1536 | ~15 GB | ~2 h |
| 12B+ | yes | 8/16 | 6e-5 → 3e-5 | 1 × 16 | 2048 | ~20 GB (sharded) | ~4 h |

Extra flags that matter:

```bash
--data data/training/dpo_pairs_v1.jsonl  # train on the proven dataset instead
--task-filter concealment                # ONE specialized adapter (Multi-LoRA)
--stage dpo                              # skip Stage A if the schema is learned
--no-4bit                                # full-precision LoRA on ≥24GB cards
--merge                                  # emit full fp16 model (needed for GGUF)
--push-hf <user>/<repo>                  # upload adapter (set HF_TOKEN first)
```

## 4. Multi-LoRA / per-capability routing (the documented lever past F≈0.61)

The v1→v4 history is the evidence: each single-adapter iteration improved
one layer at another's expense (v2: style 0.345→0.527 but cf 0.564→0.012;
v3: cf recovered, style regressed). Fix: **specialized adapters per
capability, routed by item class — no interference.** The task-tagged
dataset exists exactly for this.

Train the specialists (minutes each on 0.5B):

```bash
for T in policy concealment quotes canon; do
  python scripts/training/train_dpo.py --base Qwen/Qwen2.5-0.5B-Instruct \
      --data data/training/dpo_pairs.jsonl --task-filter $T \
      --stage dpo --out runs/fy-0.5b-$T
done
```

Route at inference (capability is observable from the request type):

```python
from peft import PeftModel

ROUTER = {"quotes": "fy-0.5b-quotes", "concealment": "fy-0.5b-concealment",
          "canon": "fy-0.5b-canon"}          # default -> policy adapter

model = PeftModel.from_pretrained(base, "runs/fy-0.5b-policy",
                                  adapter_name="policy")
for name, path in [("quotes", "runs/fy-0.5b-quotes"),
                   ("concealment", "runs/fy-0.5b-concealment"),
                   ("canon", "runs/fy-0.5b-canon")]:
    model.load_adapter(path, adapter_name=name)

def answer(task, prompt):
    model.set_adapter(ROUTER.get(task, "policy"))
    return generate(model, prompt)   # chat-apply + generate as usual
```

Production guidance from the iteration data:

- keep the **policy** adapter as the System-1 core (its dynamic 0.894 is
  already oracle-level);
- route **style/quote/concealment** to the GLM engine (already 1.0 / 0.545)
  in the two-tier config — engine v2 + System-1 core;
- on a 3B+ base, try one adapter first: cross-capability regression is a
  capacity problem, and the ceiling may dissolve; split only if the per-task
  smoke probes diverge;
- to ship one artifact: `add_weighted_adapter` task-arithmetic merge
  (e.g. 0.5·policy + 0.2·concealment + 0.15·quotes + 0.15·canon) — trades a
  little specialization for single-file deployment.

## 5. Serve + score (the eval loop)

```bash
# a) local serving (after --merge)
python llama.cpp/convert_hf_to_gguf.py runs/fy-0.5b/merged --outfile fy-0.5b-f16.gguf
./llama-quantize fy-0.5b-f16.gguf fy-0.5b-q4km.gguf Q4_K_M
./llama-server -m fy-0.5b-q4km.gguf --port 8080 -c 4096   # add --grammar-file for hard JSON

# b) score against the frozen benchmark (adds LlamaCppAdapter to the run)
python -m scripts.benchmark.fidelity_benchmark --base-url http://127.0.0.1:8080/v1

# 7B+ alternative: vLLM OpenAI-compatible server, same benchmark call
vllm serve runs/fy-7b/merged --port 8080

# c) on-GPU scoring without any local server (Kaggle kernels):
#    training/kaggle/eval_merged.py and training/kaggle/bench_finetuned.py
```

Read from the summary: `F`, `holdout_mean`, per-layer scores. Check against
§0 — especially dynamic ≥0.85 and no collapsed layer. The harness logs every
run into the DB and refreshes `docs/benchmark_results.json`.

## 6. Upload the trained model (pick any)

**Hugging Face Hub** (best for reuse):

```bash
export HF_TOKEN=hf_...
python scripts/training/train_dpo.py --base Qwen/Qwen2.5-0.5B-Instruct \
    --out runs/fy-0.5b --push-hf <your-name>/fangyuan-0.5b-lora
```

**Kaggle Models:** finished kernel → Output tab → **New Model** → select the
merged/zip output. Or fetch and re-host:

```bash
kaggle kernels output <user>/fy-dpo-train-v4 -p ./out   # adapter + metrics
kaggle kernels status <user>/fy-dpo-train-v4
```

**GitHub release (v2.0-baseline):**

```bash
GH_TOKEN=...   # keep out of shell history / git
RELEASE_ID=$(curl -s -H "Authorization: Bearer $GH_TOKEN" \
  https://api.github.com/repos/ssmurfgg04-gif/fang-yuan-strategic-system/releases/tags/v2.0-baseline \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")
curl -s -H "Authorization: Bearer $GH_TOKEN" \
     -H "Content-Type: application/zip" \
     --data-binary @fangyuan-model.zip \
     "https://uploads.github.com/repos/ssmurfgg04-gif/fang-yuan-strategic-system/releases/$RELEASE_ID/assets?name=fangyuan-0.5b-model.zip"
```

## 7. Troubleshooting

| Symptom | Fix |
|---------|-----|
| CUDA OOM | lower bsz → raise GA (keep bsz·GA ≈ 8–16); ensure 4-bit engaged (`--4bit`); drop `--seq 1024` |
| T4 rejects bf16 | automatic — the trainer detects and uses fp16 on T4/P100 |
| DPO margins saturate, no learning signal | v4 lesson: make rejections closer calls, or raise β / lower LR (`--beta 0.05`) |
| One layer improves, another collapses | single-adapter interference (v2/v3 signature) → route per capability (§4) or scale base (§3) |
| Adapter doesn't change outputs | confirm you loaded `stage_b_dpo` (not `stage_a_sft`), and served the `--merge` output |
| Gemma-3 download 401/403 | accept the license on its HF page once; `export HF_TOKEN` |
| trl/transformers API drift | pin exactly as in §2/§3 (`trl==0.19.1`, `transformers>=4.46,<5`) |
| Output drifts into prose | GBNF grammar forcing the decision-JSON schema; cap `max_new_tokens` at 220 |

---

Frozen-corpus discipline: v2.x trains on the frozen 211 decisions + 553/542
pairs. To go further, use Multi-LoRA specialists (§4) or bigger bases (§3) —
not by re-opening the corpus.
