#!/usr/bin/env bash
# Clean MM-GBSA selectivity test for the iGEM binder LISDAELEAIFEADC on PfLDH vs hLDH.
# Matched to the Liu2019 Vina test (box 30, N=100, seed 42) + MM-GBSA on top-10 clusters.
set -euo pipefail
cd /home/igem/unknown_software
source ~/miniconda3/etc/profile.d/conda.sh
conda activate score-env

PFLDH=/home/igem/unknown_software/pfldh.pdb
HLDH=/home/igem/unknown_software/data/pdbs/hldh.pdb
PF_SITE="45.149 32.445 49.428"
HL_SITE="44.647 -16.74 -37.215"
SEQ=LISDAELEAIFEADC

run() {  # name receptor "site"
  local name=$1 rec=$2 site=$3
  echo "=== $(date +%H:%M:%S)  $name  $SEQ ==="
  hybridock-pep -v dock \
    --peptide "$SEQ" \
    --receptor "$rec" \
    --site $site \
    --box 30 \
    --n-samples 100 \
    --seed 42 \
    --scoring vina \
    --refine-topk 10 \
    --output-dir "runs/liu2019/$name" \
    2>&1 | tail -4
}

run lisda_pfldh "$PFLDH" "$PF_SITE"
run lisda_hldh  "$HLDH"  "$HL_SITE"
echo "=== ALL DONE $(date +%H:%M:%S) ==="
