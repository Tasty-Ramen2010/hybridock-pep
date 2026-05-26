#!/usr/bin/env bash
# run_production_calibration.sh — Tier 0.4: Production-Pose Recalibration
#
# Runs full HybriDock-Pep docking (RAPiDock + Vina + AD4) on the 6 original
# training complexes using apo receptors (peptide extracted from the PDB).
# Collects best-pose scores and recalibrates α/β parameters.
#
# Replaces the crystal-pose calibration (α=0.10 at lower bound) with one
# that reflects actual production docking performance.
#
# Prerequisites:
#   - conda activate score-env (has vina, autodock4, ADFRsuite, babel)
#   - hybridock-pep installed (pip install -e .)
#   - datasets/raw_pdbs/ has 2HWN.pdb, 1NRL.pdb, 1L2Z.pdb, 1DDV.pdb, 1A0N.pdb, 1YWI.pdb
#   - RTX 5070 for RAPiDock (CUDA required)
#
# Usage:
#   conda run --no-capture-output -n score-env bash scripts/run_production_calibration.sh
#   # Or to just recalibrate from existing runs (no GPU):
#   SKIP_DOCKING=1 bash scripts/run_production_calibration.sh
#
# Expected time: ~8 min per complex × 6 = ~50 min on RTX 5070
# Output: data/calibration_production.json, data/training_scores_production.json

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

OUTDIR="runs/calibration_production"
SKIP_DOCKING="${SKIP_DOCKING:-0}"
N_SAMPLES="${N_SAMPLES:-100}"
SEED="${SEED:-42}"
BOX_SIZE="${BOX_SIZE:-40}"

echo "=== Tier 0.4: Production-Pose Recalibration ==="
echo "Repo: $REPO"
echo "Output dir: $OUTDIR"
echo "Skip docking: $SKIP_DOCKING"
echo ""

mkdir -p "$OUTDIR"

# ---------------------------------------------------------------------------- #
# Training complexes: pdb_id, peptide_seq, receptor_file, peptide_chain
# ---------------------------------------------------------------------------- #
declare -A PEP_SEQ
declare -A PEP_CHAIN
declare -A REC_FILE

PEP_SEQ["2hwn"]="EELAWKIAKMIVSDVMQQC"
PEP_CHAIN["2hwn"]="E"
REC_FILE["2hwn"]="datasets/raw_pdbs/2HWN.pdb"

PEP_SEQ["1nrl"]="SLTERHKILHRLLQE"
PEP_CHAIN["1nrl"]="B"
REC_FILE["1nrl"]="datasets/raw_pdbs/1NRL.pdb"

PEP_SEQ["1l2z"]="SHRPPPPGHRV"
PEP_CHAIN["1l2z"]="B"
REC_FILE["1l2z"]="datasets/raw_pdbs/1L2Z.pdb"

PEP_SEQ["1ddv"]="TPPSPF"
PEP_CHAIN["1ddv"]="B"
REC_FILE["1ddv"]="datasets/raw_pdbs/1DDV.pdb"

PEP_SEQ["1a0n"]="PPRPLPVAPGSSKT"
PEP_CHAIN["1a0n"]="B"
REC_FILE["1a0n"]="datasets/raw_pdbs/1A0N.pdb"

PEP_SEQ["1ywi"]="PPPLPP"
PEP_CHAIN["1ywi"]="B"
REC_FILE["1ywi"]="datasets/raw_pdbs/1YWI.pdb"

COMPLEXES=("2hwn" "1nrl" "1l2z" "1ddv" "1a0n" "1ywi")

# ---------------------------------------------------------------------------- #
# Step 1: Compute binding site centers from crystal peptide coordinates
# ---------------------------------------------------------------------------- #
echo "Step 1: Computing binding site centers from crystal structures..."

for PDB_ID in "${COMPLEXES[@]}"; do
    REC="${REC_FILE[$PDB_ID]}"
    if [[ ! -f "$REC" ]]; then
        echo "ERROR: Receptor file not found: $REC"
        echo "  Run: rsync from Mac or re-download"
        exit 1
    fi
done

# Python script to compute site centers
python3 - << 'PYEOF'
import sys
from pathlib import Path
import json

REPO = Path.cwd()
complexes = {
    "2hwn": ("datasets/raw_pdbs/2HWN.pdb", "E"),
    "1nrl": ("datasets/raw_pdbs/1NRL.pdb", "B"),
    "1l2z": ("datasets/raw_pdbs/1L2Z.pdb", "B"),
    "1ddv": ("datasets/raw_pdbs/1DDV.pdb", "B"),
    "1a0n": ("datasets/raw_pdbs/1A0N.pdb", "B"),
    "1ywi": ("datasets/raw_pdbs/1YWI.pdb", "B"),
}

centers = {}
for pdb_id, (rec_path, pep_chain) in complexes.items():
    try:
        coords = []
        for line in Path(rec_path).read_text().splitlines():
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if len(line) < 54:
                continue
            if line[21] != pep_chain:
                continue
            atom_name = line[12:16].strip()
            if atom_name.startswith("H"):
                continue
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                coords.append([x, y, z])
            except ValueError:
                continue
        if not coords:
            print(f"WARNING: No coords for {pdb_id} chain {pep_chain}")
            continue
        import statistics
        cx = statistics.mean(c[0] for c in coords)
        cy = statistics.mean(c[1] for c in coords)
        cz = statistics.mean(c[2] for c in coords)
        centers[pdb_id] = [round(cx, 2), round(cy, 2), round(cz, 2)]
        print(f"  {pdb_id}: center=({cx:.2f}, {cy:.2f}, {cz:.2f}) ({len(coords)} atoms)")
    except Exception as e:
        print(f"ERROR: {pdb_id}: {e}")
        sys.exit(1)

# Save centers for use in docking loop
json.dump(centers, open("runs/calibration_production/site_centers.json", "w"), indent=2)
print(f"\nSaved {len(centers)} centers to runs/calibration_production/site_centers.json")
PYEOF

# ---------------------------------------------------------------------------- #
# Step 2: Run production docking for each complex (unless SKIP_DOCKING=1)
# ---------------------------------------------------------------------------- #

if [[ "$SKIP_DOCKING" == "1" ]]; then
    echo ""
    echo "Step 2: SKIP_DOCKING=1 — skipping docking, using existing runs/"
else
    echo ""
    echo "Step 2: Running production docking for ${#COMPLEXES[@]} complexes..."
    echo "  N_SAMPLES=$N_SAMPLES, SEED=$SEED, BOX_SIZE=$BOX_SIZE"
    echo ""

    CENTERS_JSON="runs/calibration_production/site_centers.json"

    for PDB_ID in "${COMPLEXES[@]}"; do
        PEP="${PEP_SEQ[$PDB_ID]}"
        REC="${REC_FILE[$PDB_ID]}"
        RUN_OUT="$OUTDIR/$PDB_ID"

        # Skip if already completed (has ranked_poses.csv with data)
        if [[ -f "$RUN_OUT/ranked_poses.csv" ]]; then
            ROW_COUNT=$(wc -l < "$RUN_OUT/ranked_poses.csv" | tr -d ' ')
            if [[ "$ROW_COUNT" -gt 1 ]]; then
                echo "  [$PDB_ID] Skipping — ranked_poses.csv exists with $((ROW_COUNT-1)) poses"
                continue
            fi
        fi

        # Get site center from JSON
        SITE=$(python3 -c "
import json
c = json.load(open('$CENTERS_JSON'))
site = c['$PDB_ID']
print(f'{site[0]} {site[1]} {site[2]}')
")

        echo "  [$PDB_ID] Docking peptide $PEP"
        echo "    Receptor: $REC"
        echo "    Site: $SITE, Box: $BOX_SIZE Å"

        mkdir -p "$RUN_OUT"

        hybridock-pep dock \
            --peptide "$PEP" \
            --receptor "$REC" \
            --site $SITE \
            --box "$BOX_SIZE" \
            --n-samples "$N_SAMPLES" \
            --seed "$SEED" \
            --scoring vina,ad4 \
            --output-dir "$RUN_OUT" \
            2>&1 | tee "$RUN_OUT/dock.log"

        # Check success
        if [[ -f "$RUN_OUT/ranked_poses.csv" ]]; then
            POSE_COUNT=$(tail -n +2 "$RUN_OUT/ranked_poses.csv" | wc -l | tr -d ' ')
            echo "    ✓ $POSE_COUNT poses generated"
        else
            echo "    ✗ ERROR: ranked_poses.csv not created"
            echo "    Check: $RUN_OUT/dock.log"
        fi
        echo ""
    done
fi

# ---------------------------------------------------------------------------- #
# Step 3: Collect scores and recalibrate
# ---------------------------------------------------------------------------- #
echo ""
echo "Step 3: Collecting scores from ranked_poses.csv..."

python3 - << 'PYEOF'
import json
import sys
from pathlib import Path

OUTDIR = Path("runs/calibration_production")
COMPLEXES = ["2hwn", "1nrl", "1l2z", "1ddv", "1a0n", "1ywi"]

import csv
scores = {}
missing = []

for pdb_id in COMPLEXES:
    ranked_path = OUTDIR / pdb_id / "ranked_poses.csv"
    if not ranked_path.exists():
        print(f"  MISSING: {ranked_path}")
        missing.append(pdb_id)
        continue

    rows = list(csv.DictReader(ranked_path.open()))
    if not rows:
        print(f"  EMPTY: {ranked_path}")
        missing.append(pdb_id)
        continue

    best = rows[0]
    entry = {
        "vina_score": float(best["vina_score"]) if best.get("vina_score") else None,
        "ad4_score": float(best["ad4_score"]) if best.get("ad4_score") else None,
        "n_contact_residues": int(best["n_contact_residues"]) if best.get("n_contact_residues") else 0,
    }

    if entry["vina_score"] is None or entry["ad4_score"] is None:
        print(f"  INVALID SCORES: {pdb_id}: {best}")
        missing.append(pdb_id)
        continue

    scores[pdb_id] = entry
    print(f"  {pdb_id}: vina={entry['vina_score']:.2f}  ad4={entry['ad4_score']:.2f}  contacts={entry['n_contact_residues']}")

if missing:
    print(f"\nERROR: Missing/invalid scores for {missing}")
    print("Re-run docking for these complexes (unset SKIP_DOCKING or set to 0).")
    sys.exit(1)

scores_path = Path("data/training_scores_production.json")
scores_path.write_text(json.dumps(scores, indent=2))
print(f"\nSaved {len(scores)} scores to {scores_path}")
PYEOF

echo ""
echo "Step 4: Recalibrating α and β with production scores..."

hybridock-pep calibrate \
    --training-csv data/training_complexes.csv \
    --scores-json data/training_scores_production.json \
    --output data/calibration_production.json \
    --verbose

echo ""
echo "Step 5: Comparing before/after..."

python3 - << 'PYEOF'
import json

try:
    old = json.load(open("data/calibration.json"))
    new = json.load(open("data/calibration_production.json"))
    print(f"BEFORE (crystal-pose, n={old.get('n_complexes',6)}):")
    print(f"  α = {old['alpha']:.3f}  β = {old.get('beta',0.0):.3f}  r = {old.get('pearson_r',0.0):.3f}")
    print(f"\nAFTER (production-pose, n={new.get('n_complexes',6)}):")
    print(f"  α = {new['alpha']:.3f}  β = {new.get('beta',0.0):.3f}  r = {new.get('pearson_r',0.0):.3f}")

    if new['alpha'] <= 0.10:
        print("\n⚠️  WARNING: α still at lower bound (0.10)")
        print("   Cause: RAPiDock poses not reproducing binding → low correlation")
        print("   Action: Continue to Tier 1.3 (score 284 calibration entries)")
    elif new['alpha'] >= 1.5:
        print("\n⚠️  WARNING: α at upper bound (1.5) — possible overcorrection")
        print("   Check individual vina/ad4 scores for outliers")
    else:
        print(f"\n✓ α moved from {old['alpha']:.3f} → {new['alpha']:.3f} (expected: 0.3–0.9)")
        print("  Calibration is now based on production docking, not crystal poses")
except Exception as e:
    print(f"Error reading calibration files: {e}")
PYEOF

echo ""
echo "=== Done: Tier 0.4 Complete ==="
echo ""
echo "Next steps:"
echo "  1. Review calibration_production.json — if α > 0.10, promote to calibration.json:"
echo "     cp data/calibration.json data/calibration_legacy_6complex.json"
echo "     cp data/calibration_production.json data/calibration.json"
echo ""
echo "  2. Run Tier 1.3 for 284-entry calibration:"
echo "     conda run -n score-env python scripts/score_calibration_set.py \\"
echo "         --workers 8 --output-json data/training_scores_full.json"
echo ""
echo "  3. Commit:"
echo "     git add data/calibration_production.json data/training_scores_production.json"
echo '     git commit -m "feat(calibration): production-pose recalibration Tier 0.4"'
