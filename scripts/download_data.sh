#!/usr/bin/env bash
# Download ArBanking77 (CC BY-SA 4.0, SinaLab, Birzeit University) pinned to one commit,
# so the raw data can never change under us. Afterwards: `dvc add data/raw`.
set -euo pipefail

COMMIT=2e3a5639e254bc28828ed0af6d3059d64d3b97fc
BASE="https://raw.githubusercontent.com/SinaLab/ArBanking77/${COMMIT}"
OUT=data/raw
FILES=(
  Banking77_full_corpus.csv
  Banking77_intents.csv
  Banking77_Arabized_MSA_PAL_train.csv
  Banking77_Arabized_MSA_PAL_val.csv
  Banking77_Arabized_MSA_test.csv
  Banking77_Arabized_PAL_test.csv
  Banking77_Arabized_Saudi_test.csv
  Banking77_Arabized_Moroccan_test.csv
  Banking77_Arabized_Tunisian_test.csv
)

mkdir -p "$OUT"
for f in "${FILES[@]}"; do
  curl -sSfL "${BASE}/data/${f}" -o "${OUT}/${f}"
done
curl -sSfL "${BASE}/LICENSE" -o "${OUT}/LICENSE"
echo "$COMMIT" > "${OUT}/SOURCE_COMMIT"
echo "Downloaded ${#FILES[@]} files from SinaLab/ArBanking77@${COMMIT:0:7} into ${OUT}/"
