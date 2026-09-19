#!/usr/bin/env bash
# ADCP prep for the leak-free RecentSet benchmark: receptor .pdbqt + a 30 A docking box
# centred on the crystal peptide (pocket-given, the same information RAPiDock local gets).
# CPU-only. Usage: adcp_prep.sh [workers]   (default 3; keep cores free for the GPU arms)
set -u
R=/home/igem/unknown_software
A=/home/igem/ADFRsuite_x86_64Linux_1.0/bin
PY=$R/../miniconda3/envs/rapidock/bin/python
OUT=$R/runs/adcp
W=${1:-3}
mkdir -p "$OUT"

$PY - <<'PYEOF' > /tmp/claude-1000/adcp_jobs.tsv
import csv
for r in csv.DictReader(open("/home/igem/unknown_software/data/bench_recentset_heldout.csv")):
    print("\t".join([r["name"], r["receptor"], r["peptide_pdb"], r["seq"]]))
PYEOF
echo "jobs: $(wc -l < /tmp/claude-1000/adcp_jobs.tsv)"

prep_one() {
  # Every path must be defined INSIDE this function: `export -f prep_one` exports the
  # function but NOT the parent's variables, so a PY inherited from the parent script is
  # empty in the worker -- which silently produced no box numbers and failed all 345.
  R=/home/igem/unknown_software
  A=/home/igem/ADFRsuite_x86_64Linux_1.0/bin
  PY=/home/igem/miniconda3/envs/rapidock/bin/python
  OUT=$R/runs/adcp
  IFS=$'\t' read -r NAME REC PEP SEQ <<< "$1"
  d="$OUT/$NAME"; mkdir -p "$d"
  [ -s "$d/tgt.trg" ] && { echo "  $NAME: already prepped"; return 0; }
  "$A/prepare_receptor" -r "$REC" -o "$d/rec.pdbqt" >"$d/prep.log" 2>&1 || {
    echo "  $NAME: prepare_receptor FAILED"; return 1; }
  # Box sized to THIS peptide, not a fixed 30 A cube. A fixed cube cannot contain the
  # longer peptides in this set (25 of 345 exceed it; 7kei's half-extent is 28.7 A vs the
  # box's 15 A half-size), which would handicap ADCP for reasons unrelated to its search.
  # size = 2 * max half-extent + 8 A pad, floored at 30 A so small peptides keep a normal box.
  # Sizing is done in Python, not awk: an awk version of this mis-parsed the element
  # column and produced a 283 A box for 7kei (true half-extent 28.7 A). Same code path
  # as the validated fairness check.
  # Four separate fields: `read -r C S` would put only cx in C and "cy cz size" in S,
  # which scrambled agfr's box arguments on the previous run.
  read -r CX CY CZ S SP <<< "$("$PY" - "$PEP" <<'PYEOF'
import sys
xs=[];ys=[];zs=[]
for l in open(sys.argv[1]):
    if l.startswith(("ATOM","HETATM")) and l[76:78].strip() != "H":
        try: xs.append(float(l[30:38])); ys.append(float(l[38:46])); zs.append(float(l[46:54]))
        except ValueError: pass
cx,cy,cz = sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)
h = max(max(abs(v-c) for v in a) for a,c in ((xs,cx),(ys,cy),(zs,cz)))
size = max(30, round(2*h+8))
# agfr builds affinity grids over the whole box at `spacing`, so RAM scales as
# (size/spacing)**3. A 65 A box at the 0.375 A default took 20.9 GB and nearly exhausted
# the machine. Hold points-per-axis ~<=100 by coarsening spacing on big boxes only:
# 30 A keeps the 0.375 default; 65 A gets 0.65. Affects just 7 of 345 complexes (the
# longest peptides), which get a coarser ADCP search grid -- a small handicap to ADCP that
# must be disclosed, not hidden, when reporting those complexes.
spacing = max(0.375, size/100.0)
print(f"{cx:.2f} {cy:.2f} {cz:.2f} {size:.0f} {spacing:.3f}")
PYEOF
)"
  if [ -z "${S:-}" ]; then echo "  $NAME: box sizing FAILED"; return 1; fi
  ( cd "$d" && "$A/agfr" -r rec.pdbqt -b user "$CX" "$CY" "$CZ" "$S" "$S" "$S" \
       -s "$SP" -o tgt >>prep.log 2>&1 )
  [ -s "$d/tgt.trg" ] && echo "  $NAME: ok (box ${S}A spacing ${SP}A at $CX $CY $CZ)" \
    || echo "  $NAME: agfr FAILED"
}
export -f prep_one

xargs -a /tmp/claude-1000/adcp_jobs.tsv -d '\n' -P "$W" -I{} bash -c 'prep_one "$@"' _ {} \
  > "$R/logs/adcp_prep.log" 2>&1
echo "prepped ok : $(grep -c ': ok' $R/logs/adcp_prep.log)"
echo "already    : $(grep -c 'already prepped' $R/logs/adcp_prep.log)"
echo "failed     : $(grep -cE 'FAILED' $R/logs/adcp_prep.log)"
