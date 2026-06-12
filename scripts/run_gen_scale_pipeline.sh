#!/bin/bash
# Scaled augmentation: 237-complex gen subset @ N=100, CONSISTENT pocket physics
# basis for bench AND gen, then augmented CV. GPU-polite (waits for PfLDH).
set -uo pipefail
cd /home/igem/unknown_software
SCORE=/home/igem/miniconda3/envs/score-env/bin/python
RAPID=/home/igem/miniconda3/envs/rapidock/bin/python
D=logs/diagnosis
LOG=logs/gen_n100/scale_pipeline.log
mkdir -p logs/gen_n100
echo "==== scale pipeline start $(date) ====" | tee -a "$LOG"

# Seed gen physics from the 60 already done (pocket basis, resumes the rest).
cp -n "$D/feats_gen_n100_physics.pkl" "$D/feats_gen250_physics.pkl" 2>/dev/null || true

# (A) bench physics on the POCKET basis — CPU, runs concurrently with GPU gen.
echo "[A bench-pocket-physics] start $(date)" | tee -a "$LOG"
$SCORE scripts/extract_physics_features.py \
  --json logs/analysis_bench300/benchmark_results.json \
  --csv  data/benchmark300.csv \
  --out-pkl "$D/feats_bench300_pocket_physics.pkl" \
  --label bench_pocket >> "$LOG" 2>&1 &
BENCH_PHYS_PID=$!

# (0) Yield GPU to any live PfLDH production dock.
while pgrep -f "runs/pfldh_lisdaeleaifeadc" >/dev/null 2>&1; do sleep 60; done
echo "[gpu] free at $(date)" | tee -a "$LOG"

# (1) Generate 237 @ N=100 (resumes; skips the 60 done). FIXED eval → correct labels.
echo "[1 gen] $(date)" | tee -a "$LOG"
$SCORE scripts/generate_confidence_poses.py \
  --input data/gen_subset_250.csv --out-dir logs/gen_n100 --n-samples 100 >> "$LOG" 2>&1
echo "[1 gen] done $(date)" | tee -a "$LOG"

# (2) Encoder feats for all 237 (fresh; load_or_extract is all-or-nothing).
echo "[2 encoder] $(date)" | tee -a "$LOG"
rm -f "$D/feats_gen250.pkl"
PYTHONPATH=$(pwd) $RAPID scripts/extract_encoder_gen_n100.py --device cuda \
  --csv data/gen_subset_250.csv --json logs/gen_n100/benchmark_results.json \
  --out "$D/feats_gen250.pkl" --tmp logs/gen_n100/_enc250_tmp >> "$LOG" 2>&1
echo "[2 encoder] done $(date)" | tee -a "$LOG"

# (3) Gen physics on POCKET basis (resumes from the 60 seed).
echo "[3 gen-physics] $(date)" | tee -a "$LOG"
$SCORE scripts/extract_physics_features.py \
  --json logs/gen_n100/benchmark_results.json \
  --csv  data/gen_subset_250.csv \
  --out-pkl "$D/feats_gen250_physics.pkl" --label gen250 >> "$LOG" 2>&1
echo "[3 gen-physics] done $(date)" | tee -a "$LOG"

# Make sure bench-pocket physics finished.
wait $BENCH_PHYS_PID 2>/dev/null || true
echo "[A bench-pocket-physics] done $(date)" | tee -a "$LOG"

# (4) Augmented CV — CONSISTENT pocket basis for bench AND gen.
echo "[4 augmented_cv pocket-basis] $(date)" | tee -a "$LOG"
PYTHONPATH=$(pwd) $RAPID scripts/augmented_cv.py \
  --bench-phys "$D/feats_bench300_pocket_physics.pkl" \
  --gen-enc "$D/feats_gen250.pkl" \
  --gen-phys "$D/feats_gen250_physics.pkl" \
  --gen-json logs/gen_n100/benchmark_results.json >> "$LOG" 2>&1
echo "==== scale pipeline done $(date) ====" | tee -a "$LOG"
