#!/usr/bin/env bash
# Test Liu et al. 2019 (liu2019.pdf) peptide-aptamer claims under our Vina pipeline.
# P1 (original epitope) and Pf_P1 (their "optimized" peptide) on PfLDH vs hLDH,
# matched to the cached LISDAELEAIFEADC runs (same receptors/sites/box/N/seed).
set -euo pipefail
cd /home/igem/unknown_software
source ~/miniconda3/etc/profile.d/conda.sh
conda activate score-env

PFLDH=/home/igem/unknown_software/pfldh.pdb
HLDH=/home/igem/unknown_software/data/pdbs/hldh.pdb
PF_SITE="45.149 32.445 49.428"
HL_SITE="44.647 -16.74 -37.215"

run() {  # name seq receptor "site"
  local name=$1 seq=$2 rec=$3 site=$4
  echo "=== $(date +%H:%M:%S)  $name  $seq ==="
  hybridock-pep -v dock \
    --peptide "$seq" \
    --receptor "$rec" \
    --site $site \
    --box 30 \
    --n-samples 100 \
    --seed 42 \
    --scoring vina \
    --output-dir "runs/liu2019/$name" \
    2>&1 | tail -3
}

run pfp1_pfldh KITTTDEEVEGIFD "$PFLDH" "$PF_SITE"
run pfp1_hldh  KITTTDEEVEGIFD "$HLDH"  "$HL_SITE"
run p1_pfldh   KITDEEVEGIFDC  "$PFLDH" "$PF_SITE"
run p1_hldh    KITDEEVEGIFDC  "$HLDH"  "$HL_SITE"

echo "=== ALL DONE $(date +%H:%M:%S) ==="
