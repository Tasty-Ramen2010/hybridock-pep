#!/usr/bin/env bash
# compare_rapidock_models.sh
#
# Runs both original and fine-tuned RAPiDock on the PepSet benchmark,
# then computes Ca RMSD (predicted vs crystal reference) and prints
# a side-by-side summary.
#
# Prerequisites:
#   - rapidock conda env is activated (or run via conda run -n rapidock)
#   - Fine-tuning has completed and rapidock_finetuned_best.pt exists
#
# Usage:
#   conda run --no-capture-output -n rapidock bash scripts/compare_rapidock_models.sh
#   OR from rapidock env:
#   bash scripts/compare_rapidock_models.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO
PEPSET_CSV="$REPO/datasets/pepset/inference_input.csv"
ORIGINAL_CKPT="$REPO/third_party/RAPiDock/train_models/CGTensorProductEquivariantModel/rapidock_local.pt"
# Prefer final checkpoint (all 50 epochs) over best (often epoch 1 if val_loss
# monitoring is broken). Fall back to best if final doesn't exist.
_FINAL="$REPO/third_party/RAPiDock_finetuned/finetune_out/rapidock_finetuned_final.pt"
_BEST="$REPO/third_party/RAPiDock_finetuned/finetune_out/rapidock_finetuned_best.pt"
if [ -f "$_FINAL" ]; then
    FINETUNED_CKPT="$_FINAL"
    echo "Using final trained checkpoint for comparison"
else
    FINETUNED_CKPT="$_BEST"
    echo "WARNING: Using best checkpoint (may be epoch 1 — not full training)"
fi
MODEL_DIR="$REPO/third_party/RAPiDock/train_models/CGTensorProductEquivariantModel"
OUT_BASE="$REPO/runs/model_comparison"

N_SAMPLES=20    # poses per complex
INFERENCE_STEPS=20

# Rebuild inference CSV if missing
if [ ! -f "$PEPSET_CSV" ]; then
    echo "Building PepSet inference CSV..."
    python "$REPO/scripts/build_pepset_inference_csv.py"
fi

if [ ! -f "$FINETUNED_CKPT" ]; then
    echo "ERROR: No fine-tuned checkpoint found (neither final nor best). Run train_lastlayer.py first."
    exit 1
fi

if [ ! -f "$PEPSET_CSV" ]; then
    echo "ERROR: PepSet inference CSV not found at $PEPSET_CSV"
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
        --model_dir "$MODEL_DIR" \
        --output_dir "$out_dir" \
        --N "$N_SAMPLES" \
        --inference_steps "$INFERENCE_STEPS" \
        --no_final_step_noise \
        2>&1 | tee "$out_dir/inference.log"

    echo "Output written to $out_dir"
}

run_inference "original" "$ORIGINAL_CKPT"
run_inference "finetuned" "$FINETUNED_CKPT"

echo ""
echo "=============================="
echo " Ca RMSD Comparison"
echo "=============================="

python - <<'PYEOF'
"""Compute Ca RMSD between RAPiDock top-ranked pose and PepSet crystal reference.

Uses the Kabsch algorithm for optimal superposition before computing RMSD,
so results are not sensitive to the reference frame of the output PDB.
"""
import os
import sys
from pathlib import Path
import numpy as np
import csv

repo = Path(os.environ["REPO"])
base = repo / "runs" / "model_comparison"
pepset_csv = repo / "datasets" / "pepset" / "inference_input.csv"

# Load crystal reference paths from inference_input.csv
crystal_refs = {}
with open(pepset_csv, newline="") as fh:
    for row in csv.DictReader(fh):
        crystal_refs[row["complex_name"]] = Path(row["crystal_ref"])


def parse_ca_coords(pdb_path):
    """Return (N,3) Ca coordinate array from a PDB file."""
    coords = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith(("ATOM  ", "HETATM")):
                atom = line[12:16].strip()
                if atom == "CA":
                    try:
                        coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    except ValueError:
                        pass
    return np.array(coords, dtype=np.float64) if coords else np.empty((0, 3))


def kabsch_rmsd(P, Q):
    """RMSD after optimal superposition of P onto Q (Kabsch algorithm).

    P, Q: (N, 3) arrays, same N.
    """
    if len(P) != len(Q) or len(P) == 0:
        return float("nan")
    P = P - P.mean(axis=0)
    Q = Q - Q.mean(axis=0)
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    P_rot = P @ R.T
    return float(np.sqrt(np.mean(np.sum((P_rot - Q) ** 2, axis=1))))


for label in ["original", "finetuned"]:
    out_dir = base / label
    if not out_dir.exists():
        print(f"{label}: output directory missing")
        continue

    rmsds = []
    missing = 0

    for complex_name, ref_pdb in sorted(crystal_refs.items()):
        pred_dir = out_dir / complex_name
        if not pred_dir.exists():
            missing += 1
            continue

        # Find best-ranked pose (rank1.pdb or rank1_*.pdb)
        rank1_files = sorted(pred_dir.glob("rank1*.pdb"))
        if not rank1_files:
            missing += 1
            continue

        pred_ca = parse_ca_coords(rank1_files[0])
        ref_ca = parse_ca_coords(ref_pdb)

        if len(pred_ca) == 0 or len(ref_ca) == 0:
            missing += 1
            continue

        # Trim to the shorter sequence if there's a length mismatch
        n = min(len(pred_ca), len(ref_ca))
        rmsd = kabsch_rmsd(pred_ca[:n], ref_ca[:n])
        if not np.isnan(rmsd):
            rmsds.append(rmsd)

    if rmsds:
        n = len(rmsds)
        mean_r = np.mean(rmsds)
        med_r = np.median(rmsds)
        lt2 = 100 * sum(r < 2 for r in rmsds) / n
        lt5 = 100 * sum(r < 5 for r in rmsds) / n
        print(f"{label:12s}  n={n:3d}  mean={mean_r:.2f}Å  median={med_r:.2f}Å  "
              f"<2Å={lt2:.0f}%  <5Å={lt5:.0f}%  missing={missing}")
    else:
        print(f"{label:12s}  no RMSD data (missing={missing}) — check inference.log")
PYEOF
