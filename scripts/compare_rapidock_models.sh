#!/usr/bin/env bash
# compare_rapidock_models.sh
#
# Runs both original and fine-tuned RAPiDock on the PepSet benchmark fixtures,
# then prints a side-by-side Cα RMSD summary.
#
# Prerequisites:
#   - rapidock-env is activated (or run via conda run)
#   - Fine-tuning has completed and rapidock_finetuned_best.pt exists
#   - No PyRosetta relax (--no-pyrosetta flag, or pyrosetta_utils bypassed)
#
# Usage:
#   conda run --no-capture-output -n rapidock-env bash scripts/compare_rapidock_models.sh
#   OR from rapidock-env:
#   bash scripts/compare_rapidock_models.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PEPSET_CSV="$REPO/datasets/pepset/benchmark.csv"
ORIGINAL_CKPT="$REPO/third_party/RAPiDock/train_models/CGTensorProductEquivariantModel/rapidock_local.pt"
FINETUNED_CKPT="$REPO/third_party/RAPiDock_finetuned/finetune_out/rapidock_finetuned_best.pt"
MODEL_PARAMS="$REPO/third_party/RAPiDock/train_models/CGTensorProductEquivariantModel/model_parameters.yml"
OUT_BASE="$REPO/runs/model_comparison"

N_SAMPLES=20    # poses per complex (20 is enough for a quick comparison)
INFERENCE_STEPS=20

if [ ! -f "$FINETUNED_CKPT" ]; then
    echo "ERROR: Fine-tuned checkpoint not found at $FINETUNED_CKPT"
    echo "Run train_lastlayer.py first."
    exit 1
fi

if [ ! -f "$PEPSET_CSV" ]; then
    echo "ERROR: PepSet benchmark CSV not found at $PEPSET_CSV"
    exit 1
fi

run_inference() {
    local label="$1"
    local ckpt="$2"
    local out_dir="$OUT_BASE/$label"
    mkdir -p "$out_dir"

    echo ""
    echo "=============================="
    echo " Running: $label"
    echo " Checkpoint: $ckpt"
    echo "=============================="

    python "$REPO/third_party/RAPiDock/inference.py" \
        --protein_peptide_csv "$PEPSET_CSV" \
        --ckpt "$ckpt" \
        --model_parameters_path "$MODEL_PARAMS" \
        --output_dir "$out_dir" \
        --samples_per_complex "$N_SAMPLES" \
        --inference_steps "$INFERENCE_STEPS" \
        --no_final_step_noise \
        --no_pyrosetta \
        2>&1 | tee "$out_dir/inference.log"

    echo "Output written to $out_dir"
}

run_inference "original" "$ORIGINAL_CKPT"
run_inference "finetuned" "$FINETUNED_CKPT"

echo ""
echo "=============================="
echo " RMSD Comparison"
echo "=============================="
python - <<'PYEOF'
import os, sys
from pathlib import Path
import numpy as np

repo = Path(os.environ.get("REPO", Path(__file__).parent.parent))
base = repo / "runs" / "model_comparison"

for label in ["original", "finetuned"]:
    out_dir = base / label
    if not out_dir.exists():
        print(f"{label}: output directory missing")
        continue
    rmsds = []
    for complex_dir in sorted(out_dir.iterdir()):
        if not complex_dir.is_dir():
            continue
        # Look for RMSD summary files written by inference.py
        for f in complex_dir.glob("*.txt"):
            for line in f.read_text().splitlines():
                if "rmsd" in line.lower() or "RMSD" in line:
                    try:
                        val = float(line.split()[-1])
                        rmsds.append(val)
                    except ValueError:
                        pass
    if rmsds:
        print(f"{label:12s}  n={len(rmsds):3d}  mean_RMSD={np.mean(rmsds):.2f}Å  "
              f"median={np.median(rmsds):.2f}Å  "
              f"<2Å={100*sum(r<2 for r in rmsds)/len(rmsds):.0f}%  "
              f"<5Å={100*sum(r<5 for r in rmsds)/len(rmsds):.0f}%")
    else:
        print(f"{label:12s}  no RMSD data found in output (check inference.log)")
PYEOF
REPO="$REPO"
