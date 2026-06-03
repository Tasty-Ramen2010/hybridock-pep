#!/usr/bin/env bash
# Re-run the 3 failed complexes from the Jun 3 smoke test with a box
# sized from actual pose extent (not crystal extent). The crystal-derived
# boxes (35–43 Å) were 2–3× too small to contain RAPiDock's pose spread on
# extended binding sites (3DAB MDM2-like, 3EG6 WD40, 3TWR ankyrin repeat).
set -uo pipefail

export PATH="/home/igem/ADFRsuite_x86_64Linux_1.0/bin:/home/igem/miniconda3/envs/score-env/bin:/home/igem/miniconda3/bin:/usr/local/bin:/usr/bin:/bin"

ROOT="/home/igem/unknown_software"
LOG="$ROOT/logs/smoketest_jun02_rerun.log"
SUMMARY="$ROOT/runs/smoketest_jun02/run_summary.csv"
CALIBRATION="$ROOT/data/calibration_v1_2_production_entropy.json"
HDP="/home/igem/miniconda3/envs/score-env/bin/hybridock-pep"

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

echo ""
echo "================================================================"
echo "Failed-3 re-run: $(date -Iseconds)"
echo "================================================================"

# pdb_id  peptide  site_x site_y site_z  box (bigger than pose extent)
read -r -d '' RERUNS << 'EOF' || true
3DAB SQETFSDLWKLL 4.497 23.078 35.681 100
3EG6 GSARAEVHLRKS -15.024 -30.676 -6.803 90
3TWR LPHLQRSPPDGQSFR 2.737 7.155 25.203 125
EOF

while IFS=' ' read -r pdb peptide sx sy sz box; do
    [ -z "$pdb" ] && continue
    out_dir="$ROOT/runs/smoketest_jun02/$pdb"
    recep="$ROOT/runs/smoketest_jun02/inputs/${pdb}_receptor.pdb"
    # Wipe any stale partial scoring artifacts but keep poses_minimized
    rm -rf "$out_dir/poses_scored" "$out_dir/receptor.pdbqt" \
           "$out_dir/poses_pdbqt" "$out_dir/ranked_poses.csv" \
           "$out_dir/run_metadata.json"

    echo ""
    echo "=== $pdb (pep=$peptide, box=${box}Å) ==="
    echo "  start: $(date -Iseconds)"

    t0=$(date +%s)
    "$HDP" dock \
        --peptide "$peptide" \
        --receptor "$recep" \
        --site "$sx" "$sy" "$sz" \
        --box "$box" \
        --seed 42 \
        --output-dir "$out_dir" \
        --calibration "$CALIBRATION" \
        --input-poses "$out_dir/poses_minimized"
    code=$?
    t1=$(date +%s)
    elapsed=$((t1 - t0))

    n_poses=0
    if [ -f "$out_dir/ranked_poses.csv" ]; then
        n_poses=$(($(wc -l < "$out_dir/ranked_poses.csv") - 1))
    fi
    echo "  finish: $(date -Iseconds) (exit=$code, ${elapsed}s, $n_poses poses)"
    # Append a separate "rerun" entry to summary so the original failed row
    # is preserved for forensics
    echo "${pdb}_rerun,test10,$peptide,—,$code,$elapsed,$n_poses" >> "$SUMMARY"
done <<< "$RERUNS"

echo ""
echo "Re-run complete: $(date -Iseconds)"
