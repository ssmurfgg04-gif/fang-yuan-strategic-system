#!/usr/bin/env bash
# Fetch key Reverend Insanity Fandom wiki pages via z-ai page_reader (CC BY-SA source, attributed).
set -u
OUT="/home/z/my-project/fang-yuan-system/data/raw/fandom"
mkdir -p "$OUT"

fetch_page() {
  local key="$1"; local url="$2"
  if [ -s "$OUT/${key}.json" ]; then echo "[skip] ${key}"; return; fi
  echo "[page] ${key} -> ${url}"
  z-ai function -n page_reader -a "{\"url\": \"$url\"}" -o "$OUT/${key}.json" >/dev/null 2>&1 || echo "  !! failed: $key"
  sleep 1
}

BASE="https://reverend-insanity.fandom.com/wiki"
fetch_page fang_yuan            "$BASE/Fang_Yuan"
fetch_page bai_ning_bing        "$BASE/Bai_Ning_Bing"
fetch_page spring_autumn_cicada "$BASE/Spring_Autumn_Cicada"
fetch_page gu_concept           "$BASE/Gu"
fetch_page aperture             "$BASE/Aperture"
fetch_page primeval_essence     "$BASE/Primeval_Essence"
fetch_page gu_yue_clan          "$BASE/Gu_Yue_Clan"
fetch_page fang_zheng           "$BASE/Gu_Yue_Fang_Zheng"
fetch_page shang_yan_fei        "$BASE/Shang_Yan_Fei"
fetch_page heavenly_court       "$BASE/Heavenly_Court"
fetch_page venerable            "$BASE/Venerable"
fetch_page regions              "$BASE/Five_Regions"
fetch_page liquor_worm          "$BASE/Liquor_Worm"
fetch_page thirty_years_east    "$BASE/Thirty_Years_East_of_the_River_Gu"
fetch_page rank                 "$BASE/Rank"
fetch_page gu_refinement        "$BASE/Gu_Refinement"
fetch_page hu_immortal          "$BASE/Gu_Immortal"
fetch_page blessed_land         "$BASE/Blessed_Land"

echo "done:"
ls -la "$OUT" | grep -c "\.json"
