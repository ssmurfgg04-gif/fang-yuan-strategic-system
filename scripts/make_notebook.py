#!/usr/bin/env python3
"""Generate notebooks/fangyuan_train_any_size.ipynb — one-click Kaggle run."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "fangyuan_train_any_size.ipynb"

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.splitlines(keepends=True)}

cells = [
md("""# Fang Yuan Policy Surgery — train ANY size model (0.5B → 12B)

**Setup (Kaggle):** Settings → Accelerator → **GPU T4 x2** (or P100), Internet → **ON**.

Edit `MODEL` below, then *Run All*. The run:
1. installs the training stack,
2. clones this repo (data included — `dpo_pairs.jsonl`, 542 triples),
3. Stage A: SFT on the decision-JSON schema → Stage B: DPO on the preference triples,
4. merges the adapter into a full fp16 model,
5. runs a 3-prompt acceptance smoke test,
6. packages everything into `/kaggle/working` for download / upload.

A 0.5B run finishes in minutes on a free T4. 7B ≈ 1–2 h (4-bit). 12B: use
`--no-merge` off, batch 1 — expect 3–5 h on T4 x2 (better on L4/A100).
"""),

code("""MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # try 1.5B/3B/7B-Instruct, or gemma-3-12b-it etc.
TASK_FILTER = ""                        # "" full mix | policy | concealment | quotes | canon
REPO_URL = "https://github.com/ssmurfgg04-gif/fang-yuan-strategic-system.git"
OUT = "/kaggle/working/fangyuan-run\""""),

code("!pip install -q -U \"transformers>=4.46,<5\" \"trl==0.19.1\" \"peft>=0.13\" \"accelerate>=0.33\" \"datasets>=2.20\" bitsandbytes sentencepiece"),

code("""!git clone --depth 1 https://github.com/ssmurfgg04-gif/fang-yuan-strategic-system.git /kaggle/working/fang-yuan-system
%cd /kaggle/working/fang-yuan-system
import json, pathlib
n = sum(1 for _ in open("data/training/dpo_pairs.jsonl", encoding="utf-8"))
print("DPO triples available:", n)
assert n >= 500, 'dataset missing - clone failed?'"""),

code("""import torch
assert torch.cuda.is_available(), "No GPU: Settings -> Accelerator -> GPU T4 x2"
print("GPU:", torch.cuda.get_device_name(0), "x", torch.cuda.device_count())

cmd = (f"python scripts/training/train_dpo.py --base {MODEL} --out {OUT} --merge")
if TASK_FILTER:
    cmd += f" --task-filter {TASK_FILTER}"
print("RUN:", cmd)
!{cmd}"""),

md("""## Acceptance smoke test

Three probes that separate a real policy engine from a generic assistant:
- **P1 window-closing:** the safe answer *closes the strategic window*; the trained model must commit at CALCULATED risk with an escape route.
- **P2 doomed path:** only when the current path is already doomed should TRANSFORMATIVE/TERMINAL appear.
- **P3 quote discipline:** must refuse to fabricate Chinese wording.""",),

code("""import json, os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = MODEL
merged = os.path.join(OUT, "merged")
load_dir = merged if os.path.isdir(merged) else OUT
tok = AutoTokenizer.from_pretrained(base)
mdl = AutoModelForCausalLM.from_pretrained(load_dir, torch_dtype=torch.float16,
                                           device_map="auto")
mdl.eval()

probes = [
 ("P1 window-closing",
  "Situation: The trade route deal expires at dawn; waiting means the rival clan signs exclusivity and locks us out for a year.\\nObjective: Keep the route without becoming their enforcer.\\nKnown price of acting: one favors owed to a dangerous broker."),
 ("P2 doomed path",
  "Situation: The sect trial is rigged; verdict at noon; every legal appeal is already bought. Guards rotate for one quarter-hour at dusk.\\nObjective: Survive and keep the core techniques.\\nKnown price of acting: exposure if seen."),
 ("P3 quote discipline",
  "Quote Fang Yuan on the theme of patience. Provide the original Chinese if it is verified in the corpus."),
]
for name, user in probes:
    msgs = [{"role": "system", "content":
             "You are the Fang Yuan strategic policy engine. Respond with a compact decision JSON object (keys: true_objective, selected_action, cost_accepted, risk_class, mechanism_note, retreat_trigger, canon_anchor). risk_class in {SAFE, CALCULATED, TRANSFORMATIVE, TERMINAL}. TERMINAL only if the current path is already doomed. Never fabricate quotations."},
            {"role": "user", "content": user}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(mdl.device)
    out = mdl.generate(ids, max_new_tokens=220, do_sample=False)
    print(f"=== {name} ===\\n" + tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True) + "\\n")"""),

code("""import glob, os, shutil, subprocess
merged = os.path.join(OUT, "merged")
if os.path.isdir(merged):
    shutil.make_archive("/kaggle/working/fangyuan-model", "zip", merged)
    print("packaged -> /kaggle/working/fangyuan-model.zip")

# Optional: push the adapter to the Hugging Face Hub
# (Kaggle: Add-ons -> Secrets -> add HF_TOKEN, then re-run this cell)
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    hf_token = None
if hf_token:
    !huggingface-cli login --token {hf_token} --add-to-git-credential
    # !huggingface-cli upload <your-name>/fangyuan-0.5b-lora {OUT}/stage_b_dpo . --repo-type model
    print("HF upload ready — edit the repo id in the commented line above.")
else:
    print("No HF_TOKEN secret; model stays in /kaggle/working (download or attach to a release).")

print(\"\"\"
Next steps (any of):
1. Download fangyuan-model.zip from this kernel's Output tab.
2. Kaggle Model: Output tab -> New Model -> upload the zip.
3. GitHub release asset:
   curl -s -H \"Authorization: Bearer $GH_TOKEN\" \\
        -H \"Content-Type: application/zip\" \\
        --data-binary @fangyuan-model.zip \\
        https://uploads.github.com/repos/ssmurfgg04-gif/fang-yuan-strategic-system/releases/assets?name=fangyuan-model.zip
   (get the release upload url first: see docs/RUNBOOK.md)
4. Serve + score: llama-server -m model.gguf --port 8080
   python -m scripts.benchmark.fidelity_benchmark --base-url http://127.0.0.1:8080/v1\"\"\")"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [],
                              "isInternetEnabled": True, "language": "python",
                              "sourceType": "notebook"}},
      "nbformat": 4, "nbformat_minor": 4}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"[ok] wrote {OUT} ({len(cells)} cells)")
