#!/usr/bin/env bash
# Deep web research pass #1 — Reverend Insanity strategic-model project
# Runs structured web searches via z-ai CLI and stores raw JSON under data/raw/
set -u
OUT="/home/z/my-project/fang-yuan-system/data/raw"
mkdir -p "$OUT"

run_q() {
  local key="$1"; shift
  local query="$1"; shift
  local num="${1:-6}"
  if [ -s "$OUT/search_${key}.json" ]; then echo "[skip] ${key}"; return; fi
  echo "[search] ${key}"
  z-ai function -n web_search -a "{\"query\": \"$query\", \"num\": $num}" -o "$OUT/search_${key}.json" >/dev/null 2>&1 || echo "  !! failed: $key"
  sleep 1
}

# --- Corpus & canon facts ---
run_q ri_volumes      "Reverend Insanity 蛊真人 volumes list total chapters 2334 banned 2019" 8
run_q ri_arcs         "Reverend Insanity story arcs Qing Mao Mountain Shang Clan Northern Plains Imperial Court" 8
run_q ri_fangyuan     "Fang Yuan Reverend Insanity character Spring Autumn Cicada rebirth 500 years demonic path" 8
run_q ri_quotes       "Fang Yuan Reverend Insanity famous quotes demonic path eternal life original Chinese" 8
run_q ri_gumechanics  "Reverend Insanity Gu system aperture primeval essence ranks Gu Master Gu Immortal dao marks" 8
run_q ri_venerables   "Reverend Insanity Ten Venerables list Primordial Origin Star Constellation Giant Sun Thieving Heaven" 8
run_q ri_factions     "Reverend Insanity five regions Heavenly Court Longevity Heaven Gu Yue clan factions" 8

# --- Legal / provenance landscape ---
run_q legal_bartz     "Bartz v Anthropic fair use pirated books LLM training ruling" 6
run_q legal_kadrey    "Kadrey v Meta shadow library fair use ruling books" 6
run_q fandom_license  "Fandom wiki content license CC BY-SA attribution terms textapi" 6

# --- Engineering: small-model policy training ---
run_q qwen_lora       "Qwen2.5-0.5B LoRA fine-tune best practices learning rate dataset size 2025" 8
run_q dpo_grpo        "DPO GRPO preference tuning small LLM policy behavior dataset size" 8
run_q rag_eval        "RAGAS RAG evaluation metrics faithfulness recall precision context" 6
run_q mcts_llm        "LLM agent Monte Carlo tree search beam rollout strategic decision benchmark" 8
run_q small_model_json "small LLM constrained JSON output decision schema grammar llama.cpp GBNF" 6

# --- Lawful text sources ---
run_q ctext_api       "ctext.org Chinese Text Project API terms of use no scraping" 6
run_q wikisource_pd   "zh.wikisource 道德經 孫子兵法 韓非子 public domain full text" 6

echo "done. files:"
ls -la "$OUT" | grep search_ | wc -l
