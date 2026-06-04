#!/bin/bash
# Full n=100 gen-subset pipeline, GPU-polite:
#   wait for PfLDH production dock → generate (n=100) → encoder feats → physics
#   feats → augmented CV. Logs every stage to logs/gen_n100/pipeline.log.
set -uo pipefail
cd /home/igem/unknown_software
SCORE=/home/igem/miniconda3/envs/score-env/bin/python
RAPID=/home/igem/miniconda3/envs/rapidock/bin/python
LOG=logs/gen_n100/pipeline.log
mkdir -p logs/gen_n100
echo "==== pipeline start $(date) ====" | tee -a "$LOG"

# 0. Yield GPU to any live PfLDH production dock.
while pgrep -f "runs/pfldh_lisdaeleaifeadc" >/dev/null 2>&1; do sleep 60; done
echo "[gpu] free at $(date)" | tee -a "$LOG"

# 1. Generate n=100 poses (resumes; skips already-done complexes).
echo "[1/4 gen] $(date)" | tee -a "$LOG"
$SCORE scripts/generate_confidence_poses.py \
  --input data/gen_subset_n100.csv --out-dir logs/gen_n100 --n-samples 100 \
  >> "$LOG" 2>&1
echo "[1/4 gen] done $(date)" | tee -a "$LOG"

# 2. Encoder (96-dim) features — GPU, frozen BN.
echo "[2/4 encoder] $(date)" | tee -a "$LOG"
PYTHONPATH=$(pwd) $RAPID scripts/extract_encoder_gen_n100.py --device cuda >> "$LOG" 2>&1
echo "[2/4 encoder] done $(date)" | tee -a "$LOG"

# 3. Physics (ref2015) features — CPU, receptors from the subset CSV.
echo "[3/4 physics] $(date)" | tee -a "$LOG"
$SCORE scripts/extract_physics_features.py \
  --json logs/gen_n100/benchmark_results.json \
  --csv  data/gen_subset_n100.csv \
  --out-pkl logs/diagnosis/feats_gen_n100_physics.pkl \
  --label gen_n100 >> "$LOG" 2>&1
echo "[3/4 physics] done $(date)" | tee -a "$LOG"

# 4. Augmented CV: does bench+gen beat bench-only on held-out bench?
echo "[4/4 augmented_cv] $(date)" | tee -a "$LOG"
PYTHONPATH=$(pwd) $RAPID scripts/augmented_cv.py >> "$LOG" 2>&1
echo "==== pipeline done $(date) ====" | tee -a "$LOG"
