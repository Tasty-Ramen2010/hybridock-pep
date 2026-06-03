#!/usr/bin/env bash
# Sequential launcher for the 17-complex full-pipeline smoke test.
# Reads runs/smoketest_jun02/run_plan.csv and runs `hybridock-pep dock` per row.
# Logs to logs/smoketest_jun02.log; per-complex output in runs/smoketest_jun02/{PDB}/.
set -uo pipefail

ROOT="/home/igem/unknown_software"
cd "$ROOT" || exit 1

PLAN="$ROOT/runs/smoketest_jun02/run_plan.csv"
LOG="$ROOT/logs/smoketest_jun02.log"
SUMMARY="$ROOT/runs/smoketest_jun02/run_summary.csv"
CALIBRATION="$ROOT/data/calibration_v1_2_production_entropy.json"
PY="/home/igem/miniconda3/envs/score-env/bin/python"
HDP="/home/igem/miniconda3/envs/score-env/bin/hybridock-pep"

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

echo ""
echo "================================================================"
echo "Smoke test launched: $(date -Iseconds)"
echo "Plan: $PLAN"
echo "Calibration: $CALIBRATION"
echo "================================================================"

# Header for summary CSV (only if not already present)
if [ ! -s "$SUMMARY" ]; then
    echo "pdb_id,set,peptide,pkd,exit_code,elapsed_sec,n_poses_scored" > "$SUMMARY"
fi

# Parse plan (skip header)
total=$(($(wc -l < "$PLAN") - 1))
i=0
while IFS=, read -r set pdb peptide pkd recep crystal sx sy sz box n_pep n_rec cluster; do
    # skip header
    [ "$set" = "set" ] && continue
    i=$((i + 1))

    out_dir="$ROOT/runs/smoketest_jun02/$pdb"
    if [ -d "$out_dir/scored_poses.csv" ] || [ -f "$out_dir/ranked_poses.csv" ]; then
        echo "[$i/$total] SKIP $pdb — already has output"
        continue
    fi

    echo ""
    echo "[$i/$total] === $pdb ($set, pep=$peptide, pKd=$pkd) ==="
    echo "  receptor: $recep"
    echo "  site: ($sx, $sy, $sz)  box: $box Å"
    echo "  output: $out_dir"
    echo "  start: $(date -Iseconds)"

    t0=$(date +%s)
    "$HDP" dock \
        --peptide "$peptide" \
        --receptor "$ROOT/$recep" \
        --site "$sx" "$sy" "$sz" \
        --box "$box" \
        --n-samples 100 \
        --seed 42 \
        --output-dir "$out_dir" \
        --calibration "$CALIBRATION"
    code=$?
    t1=$(date +%s)
    elapsed=$((t1 - t0))

    n_poses=0
    if [ -f "$out_dir/ranked_poses.csv" ]; then
        n_poses=$(($(wc -l < "$out_dir/ranked_poses.csv") - 1))
    fi

    echo "  finish: $(date -Iseconds) (exit=$code, ${elapsed}s, $n_poses poses)"
    echo "$pdb,$set,$peptide,$pkd,$code,$elapsed,$n_poses" >> "$SUMMARY"
done < "$PLAN"

echo ""
echo "================================================================"
echo "All complete: $(date -Iseconds)"
echo "Summary: $SUMMARY"
echo "================================================================"
