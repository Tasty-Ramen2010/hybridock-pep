#!/usr/bin/env python3
"""Measure how long a whole prediction takes on THIS computer, so the app can say "about N minutes" and be right.

Runs the real ``hybridock-pep dock`` command (the one the web page starts) on the MDM2 site for every combination of peptide
length (8 to 25 letters) and pose count (5, 10, 25), then checks the table on points it did NOT measure and reports how far
off the estimate was. The first run only warms the caches (RAPiDock's lookup tables, the language-model weights) and is not counted.

Run time is not a straight line in length: from 13 residues the longer-peptide model is used and costs several times more per
pose, so a table of measurements, joined by lines between its points (:mod:`hybridock_pep.web.timing`), is what the page reads.

    conda activate score-env
    python3 scripts/benchmark_timing.py --machine "NVIDIA Jetson AGX Orin" \
        --out src/hybridock_pep/web/static/data/timing.json

``--seed FILE`` reuses measurements from an earlier run of this same script on this same setup (a JSON list of
``{"length": 12, "poses": 5, "seconds": 67.6}``), so an interrupted benchmark does not start over. Start the server with
``HYBRIDOCK_TIMING_FILE`` pointing at the output (the AGX launcher does this). Re-run whenever the hardware or software changes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from hybridock_pep.web.timing import estimate  # noqa: E402

RECEPTOR = REPO / "data" / "pdbs" / "1YCR_mdm2.pdb"
SITE = ("25.20", "-25.61", "-7.97")
REF_LENGTH = 12
POSES = (5, 10, 25)
#: one peptide per length. 13 is the first that uses the longer-peptide model; 15 is the project's own PfLDH peptide.
PEPTIDES = {8: "FSDLWKLL", 10: "ETFSDLWKLL", 12: "ETFSDLWKLLPE", 13: "SQETFSDLWKLLP", 14: "SQETFSDLWKLLPE", 15: "LISDAELEAIFEADC",
            16: "SQETFSDLWKLLPENN", 18: "SQETFSDLWKLLPENNVL", 20: "SQETFSDLWKLLPENNVLSP", 25: "SQETFSDLWKLLPENNVLSPLPSQA"}
GRID_LENGTHS = (8, 10, 12, 13, 15, 16, 20, 25)
#: points left out of the table on purpose, measured afterwards to see how good the estimate is between measurements
HOLDOUT = ((14, 10), (18, 25), (10, 25))


def run_once(exe: str, peptide: str, poses: int, out: Path) -> float:
    """Wall-clock seconds for one complete dock. Raises if it fails (a timing of a failure is worthless)."""
    shutil.rmtree(out, ignore_errors=True)
    cmd = [exe, "dock", "--peptide", peptide, "--receptor", str(RECEPTOR), "--site", *SITE, "--box", "30",
           "--n-samples", str(poses), "--output-dir", str(out)]
    start = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    took = time.monotonic() - start
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-12:])
        raise SystemExit(f"dock failed ({peptide}, {poses} poses):\n{tail}")
    return took


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machine", required=True, help='what to call this computer on the page, e.g. "NVIDIA Jetson AGX Orin"')
    ap.add_argument("--out", type=Path, required=True, help="where to write the timing JSON")
    ap.add_argument("--exe", default=shutil.which("hybridock-pep") or "hybridock-pep", help="the hybridock-pep command")
    ap.add_argument("--work", type=Path, default=REPO / "runs" / "bench", help="scratch folder for the runs")
    ap.add_argument("--seed", type=Path, help="earlier measurements to reuse (JSON list of {length, poses, seconds})")
    ap.add_argument("--no-holdout", action="store_true", help="skip the check on points that were not measured")
    args = ap.parse_args()

    known: dict[tuple[int, int], float] = {}
    if args.seed and args.seed.exists():
        for row in json.loads(args.seed.read_text(encoding="utf-8")):
            known[(int(row["length"]), int(row["poses"]))] = float(row["seconds"])
    args.work.mkdir(parents=True, exist_ok=True)
    print("warm-up run (not counted)...", flush=True)
    run_once(args.exe, PEPTIDES[REF_LENGTH], 5, args.work / "warm")

    def measure(length: int, poses: int) -> float:
        if (length, poses) in known:
            t = known[(length, poses)]
            print(f"  {length:>2} letters, {poses:>2} poses: {t:6.1f} s  (reused)", flush=True)
            return t
        t = run_once(args.exe, PEPTIDES[length], poses, args.work / f"{length}_{poses}")
        known[(length, poses)] = t
        print(f"  {length:>2} letters, {poses:>2} poses: {t:6.1f} s", flush=True)
        return t

    grid = {str(L): {str(n): round(measure(L, n), 1) for n in POSES} for L in GRID_LENGTHS}
    numeric = {int(L): {int(n): t for n, t in row.items()} for L, row in grid.items()}

    holdout = []
    if not args.no_holdout:
        print("checking the estimate on points that are not in the table...", flush=True)
        for length, poses in HOLDOUT:
            predicted = estimate(numeric, length, poses)
            actual = measure(length, poses)
            holdout.append({"length": length, "poses": poses, "predicted": round(predicted, 1), "measured": round(actual, 1),
                            "error": round((predicted - actual) / actual, 3)})
            print(f"  check: {length} letters, {poses} poses: predicted {predicted:.0f} s, measured {actual:.0f} s ({(predicted - actual) / actual:+.0%})", flush=True)

    out = {"machine": args.machine, "refLength": REF_LENGTH, "grid": grid, "holdout": holdout,
           "measuredAt": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "python": sys.version.split()[0]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}\n{json.dumps(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
