# Fang Yuan Policy Surgery — Universal Training Runbook (v2.0-baseline)

> Goal: train **any** instruction model — 0.5B, 3B, 7B, or 12B — to the
> system's policy level, then verify it on the frozen benchmark and upload
> the artifact. One dataset (`dpo_pairs.jsonl`, 542 triples), one trainer
> (`scripts/training/train_dpo.py`), size presets handled automatically.

---

## 0. What "our level" means (the acceptance bar)

The fidelity benchmark (`scripts/benchmark/fidelity_benchmark.py`, 140 items,
F = 0.20O+0.15A+0.15R+0.15I+0.15P+0.10M+0.10C, 10 secret holdout items) last
logged these reference scores:

| Adapter              | F     | Holdout (10 secret items) |
|----------------------|-------|---------------------------|
| policy oracle        | 0.651 | 0.944                     |
| safe generic (bad)   | 0.329 | 0.590                     |
| reckless theatrical  | 0.219 | 0.443                     |

A trained model is at **our level** when:

1. `policy-F` > both baseline adapters (the safe and reckless caricatures) —
   the model must not have collapsed back into either one;
2. `policy-F` ≥ 0.65 (matches the policy oracle), stretch bar **F ≥ 0.829**;
3. holdout mean ≥ 0.90 (no overfitting to visible items);
4. the smoke-test probes in the Kaggle notebook behave: commits at
   CALCULATED risk with an escape route when the window is closing; goes
   TRANSFORMATIVE/TERMINAL **only** when the path is already doomed; never
   fabricates Chinese quotations.

Record every run: the harness writes `docs/benchmark_results.json` and logs
to the DB automatically.

---

## 1. The data (frozen v2 corpus — do not extend for v2.x)

| File | Records | Purpose |
|------|---------|---------|
| `data/training/dpo_pairs.jsonl` | **542** | Stage B (DPO): teaches the trade-off weights |
| `data/training/sft_decisions.jsonl` | 211 | Stage A (SFT): teaches the decision-JSON schema |
| `data/training/quote_verification.jsonl` | 22 | quote discipline (already folded into DPO pairs) |
| `data/training/canon_vs_inference.jsonl` | 22 | canon discipline (already folded into DPO pairs) |

DPO triple anatomy — `prompt` = strict-constraint scenario, `chosen` = cold
utility-optimal decision JSON, `rejected` = one of two failure families:

| pair_type | count | what the rejected response does wrong |
|-----------|-------|----------------------------------------|
| safe_moral | 171 | hedges, disclaims, postpones — closes the strategic window |
| reckless | 171 | theatrical cruelty, all-in, no escape route |
| mask_rigid | 40 | fixed identity instead of utility-selected mask |
| reveal_blade | 40 | discards the mask for pride |
| inference_as_canon | 22 | presents inference as canon |
| source_free_certainty | 22 | 100% certainty, no sources |
| sunk_cost_nearmiss | 35 | doubles down on a historically failed path |
| fabricated_chinese | 19 | invents "verbatim" Chinese quotes |
| overconfident_attribution | 19 | claims unverified wording is canon |
| classic_misattribution | 3 | cites Sunzi/Laozi/Shiji as Fang Yuan |

Every record carries a `task` tag (`policy` 377 / `concealment` 80 /
`canon` 44 / `quotes` 41) — this is what enables Multi-LoRA routing (§4).

**Anti-leak guarantee:** the builder never reads `benchmark_items`; the 10
secret holdout items cannot leak into training data.

Rebuild deterministically any time (the DB is frozen, so output is stable):

```bash
python scripts/training/build_dpo_triples.py   # rng seed 211
```

---

## 2. Path A — Kaggle, zero setup (recommended)

The notebook does everything: install → clone → SFT → DPO → merge →
smoke-test → package.

1. Go to kaggle.com → **Code** → **File → Import Notebook** and upload
   `notebooks/fangyuan_train_any_size.ipynb` (or paste it into a new notebook).
2. Notebook options: **Accelerator = GPU T4 x2**, **Internet = ON**.
3. Edit the first cell (`MODEL`, optional `TASK_FILTER`), then **Run All**.

| MODEL | Time on free T4 x2 | Notes |
|-------|--------------------|-------|
| `Qwen/Qwen2.5-0.5B-Instruct` | ~5–10 min | the default; recommended first run |
| `Qwen/Qwen2.5-1.5B-Instruct` | ~15–25 min | |
| `Qwen/Qwen2.5-3B-Instruct` | ~40–60 min | auto 4-bit |
| `Qwen/Qwen2.5-7B-Instruct` | ~1.5–2.5 h | auto 4-bit |
| `google/gemma-3-12b-it` / `Qwen/Qwen2.5-14B-Instruct` | ~3–5 h | batch 1, GA 16; accept Gemma license on HF first |

Outputs land in `/kaggle/working`:
`fangyuan-run/stage_b_dpo/` (adapter), `fangyuan-run/merged/` (full fp16
model), `fangyuan-model.zip` (packaged), plus printed upload commands.
Optional: Add-ons → Secrets → add `HF_TOKEN`, and the last cell pushes the
adapter straight to the Hugging Face Hub.

## 3. Path B — any GPU box (local machine / cloud / CI runner)

```bash
git clone https://github.com/ssmurfgg04-gif/fang-yuan-strategic-system.git
cd fang-yuan-system
pip install -U "transformers>=4.46,<5" "trl==0.19.1" "peft>=0.13" \
    "accelerate>=0.33" "datasets>=2.20" bitsandbytes sentencepiece

# 0.5B (fits anywhere, minutes)
python scripts/training/train_dpo.py --base Qwen/Qwen2.5-0.5B-Instruct --out runs/fy-0.5b --merge

# 3B (auto-switches to 4-bit NF4 QLoRA)
python scripts/training/train_dpo.py --base Qwen/Qwen2.5-3B-Instruct --out runs/fy-3b --merge

# 7B
python scripts/training/train_dpo.py --base Qwen/Qwen2.5-7B-Instruct --out runs/fy-7b --merge

# 12B-class (Gemma-3-12b / Qwen2.5-14B): batch 1 + GA 16, 4-bit
python scripts/training/train_dpo.py --base google/gemma-3-12b-it --out runs/fy-12b --merge
```

Full hyperparameter matrix (what the presets resolve to; override with flags):

| Size | 4-bit | LoRA r/α | LR SFT→DPO | bsz × GA | seq | Peak VRAM | T4×2 est. |
|------|-------|----------|------------|----------|------|-----------|-----------|
| 0.5B | no | 16/32 | 2e-4 → 1e-4 | 8 × 2 | 1024 | <6 GB | ~5 min |
| 1.5B | no | 16/32 | 1e-4 → 5e-5 | 4 × 4 | 1024 | ~9 GB | ~20 min |
| 3B | yes | 16/32 | 1e-4 → 5e-5 | 4 × 4 | 1536 | ~11 GB | ~45 min |
| 7B | yes | 16/32 | 8e-5 → 4e-5 | 2 × 8 | 1536 | ~15 GB | ~2 h |
| 12B+ | yes | 8/16 | 6e-5 → 3e-5 | 1 × 16 | 2048 | ~20 GB (sharded) | ~4 h |

Extra flags that matter:

```bash
--task-filter concealment     # train ONE specialized adapter (§4)
--stage dpo                   # skip Stage A (schema) if you already have it
--no-4bit                     # full-precision LoRA on ≥24GB cards (better for 12B)
--merge                       # also emit a full fp16 model (needed for GGUF/llama.cpp)
--push-hf <user>/<repo>       # upload the adapter (set HF_TOKEN env var first)
```

## 4. Multi-LoRA / MoE-style routing (the path past F = 0.829)

A single 0.5B adapter juggles policy, concealment, quotes and canon — the
tasks compete for the same rank-16 delta. The fix: **specialized adapters +
inference-time routing** (same dataset, no new labeling).

Train the specialists (each run is minutes on 0.5B):

```bash
for T in policy concealment quotes canon; do
  python scripts/training/train_dpo.py --base Qwen/Qwen2.5-0.5B-Instruct \
      --task-filter $T --stage dpo --out runs/fy-0.5b-$T
done
```

Route at inference — a rule-based router is enough to start (request type is
observable from the system prompt):

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

Upgrade path when rule-based routing stops being enough: train a tiny
classifier (or let the 0.5B itself emit `task` first, GBNF-constrained) and
route on its output. To ship a single artifact instead, merge adapters with
task arithmetic (weighted average, e.g. `0.5·policy + 0.2·concealment +
0.15·quotes + 0.15·canon`) via `peft` `add_weighted_adapter` — losing a
little specialization for one-file deployment.

On a 3B+ base, single-adapter cross-capability regression is much weaker —
try one adapter first (§3), and only split if the per-task smoke probes
diverge.

## 5. Serve + score (the eval loop)

```bash
# a) merge + GGUF (CPU inference / llama.cpp) — train with --merge first
python llama.cpp/convert_hf_to_gguf.py runs/fy-0.5b/merged --outfile fy-0.5b-f16.gguf
./llama-quantize fy-0.5b-f16.gguf fy-0.5b-q4km.gguf Q4_K_M
./llama-server -m fy-0.5b-q4km.gguf --port 8080 -c 4096   # add --grammar-file for hard JSON

# b) score against the frozen benchmark (adds LlamaCppAdapter to the run)
python -m scripts.benchmark.fidelity_benchmark --base-url http://127.0.0.1:8080/v1

# 7B+ alternative: vLLM OpenAI-compatible server, same benchmark call
vllm serve runs/fy-7b/merged --port 8080
```

Read from the printed summary: `F`, `holdout_mean`, and the policy-vs-safe
separation. Check them against §0. The three adapters always run together in
one pass, so you always see whether your model beats its caricatures.

## 6. Upload the trained model (pick any)

**Hugging Face Hub** (best for reuse):

```bash
export HF_TOKEN=hf_...
python scripts/training/train_dpo.py --base Qwen/Qwen2.5-0.5B-Instruct \
    --out runs/fy-0.5b --push-hf <your-name>/fangyuan-0.5b-lora
```

**Kaggle Models:** from the finished kernel → Output tab → **New Model** →
select `fangyuan-model.zip`. Or CLI: `kaggle kernels output <user>/<kernel-slug> -p .`

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
| `DPO loss ≈ 0` immediately, rewards both ~0 | beta too high for the size: `--beta 0.05`; or lr too low for 0.5B |
| Adapter doesn't change outputs | confirm you loaded `stage_b_dpo` (not `stage_a_sft`), and that `--merge` output is what you served |
| Gemma-3 download 401/403 | accept the license on its HF page once; `export HF_TOKEN` |
| trl/transformers API drift | pin exactly as in §2/§3 (`trl==0.19.1`, `transformers>=4.46,<5`) |
| Benchmark hangs | the harness also runs the simulator layers — first run takes minutes; subsequent runs are cached |
| Output drifts into prose | serve with a GBNF grammar forcing the decision-JSON schema; cap `max_new_tokens` at 220 |

---

Frozen-corpus discipline: v2.x trains on the frozen 211 decisions + 542
triples. To go further, extend behavior with Multi-LoRA specialists (§4) or
bigger bases (§3) — not by re-opening the corpus.
