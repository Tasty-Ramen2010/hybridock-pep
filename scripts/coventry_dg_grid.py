#!/usr/bin/env python
"""Score the whole Coventry grid with OUR calibrated dG model, not Rosetta ref2015.

WHY THIS RUN EXISTS.  The grid has been ranked on ref2015 interface energy throughout, and
that choice deserves more scrutiny than it got.  Against it:

  ref2015 is close to useless at ABSOLUTE peptide affinity.  On 918 leakage-free
  peptide-clustered PDBbind complexes it scores r = 0.006; our calibrated model scores
  r = 0.317.  On that task it is not a weaker scorer, it is no scorer at all.

For it, and this is the only evidence I had when I picked it:

  our dG model looked POSE-BLIND on 9CCE.  A 1.99 A co-folded pose scored -10.27 kcal/mol,
  our own 6.74 A pose scored -10.20, and the actual crystal scored -9.51 -- i.e. it ranked
  the crystal WORSE than two wrong poses, over a 0.07 kcal/mol spread.  ref2015 interface
  energy, by contrast, did separate sequences in the threading test (cognate ranked #1 in
  4 of 5 cases, by 9-12 REU).

One complex is not enough to exclude our own scorer from 324 cells, so this measures it
properly: same poses, same receptors, same best-of-24 rule, the only change is the scoring
function.  Both are then put through the identical grid metrics.

Note the two are not on the same scale and must never be compared by magnitude: ref2015 is in
Rosetta Energy Units and our model is in kcal/mol, and REU converts to kcal/mol only through a
fitted linear map.  What is comparable is the RANKING each produces.

Usage: coventry_dg_grid.py [workers]
Output: logs/coventry_dg.jsonl  (schema matches coventry_refine.jsonl: name + best_iface)
"""
from __future__ import annotations

import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
OUT = ROOT / "logs/coventry_dg.jsonl"
POSES = ROOT / "runs/coventry/hybridock_ft"
ESM = ROOT / "datasets/coventry/binders"
AF3 = ROOT / "datasets/coventry/binders_af3"
#: binders where our ESMFold model is the proven outlier (Boltz and AF3 agree against it)
BROKEN = {"n3", "n7", "pc21", "pc26"}


def _init() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(ROOT / "src"))


def score_cell(args: tuple[str, str, str]) -> dict:
    """Best-of-24 calibrated dG for one grid cell."""
    from hybridock_pep.scoring.affinity_model import predict_affinity
    from hybridock_pep.scoring.geometry_features import compute_geometry_features

    name, binder, seq = args
    rec = (AF3 if binder in BROKEN else ESM) / f"{binder}_1b1.pdb"
    vals: list[float] = []
    for pose in sorted((POSES / name).glob("rank*.pdb")):
        try:
            g = compute_geometry_features(pose, rec)
            if not g:
                continue
            v = predict_affinity(g, seq)
            if v is not None:
                vals.append(float(v))
        except Exception:  # noqa: BLE001 -- one bad pose must not kill the cell
            continue
    if not vals:
        return {"name": name, "best_iface": None, "n_scored": 0}
    vals.sort()
    return {
        "name": name,
        # keyed "best_iface" so the existing grid-metrics script reads it unchanged;
        # the value is kcal/mol from our model, NOT Rosetta REU
        "best_iface": vals[0],
        "med_iface": vals[len(vals) // 2],
        "top3_iface": sum(vals[:3]) / min(3, len(vals)),
        "n_scored": len(vals),
        "n_clean": len(vals),
        "units": "kcal/mol",
        "receptor_model": "af3" if binder in BROKEN else "esmfold",
    }


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    seqs = {r["name"]: r["sequence"]
            for r in csv.DictReader(open(ROOT / "data/coventry_targets.csv"))}
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["name"])

    jobs = []
    for d in sorted(POSES.glob("*__*")):
        if d.name in done or not list(d.glob("rank*.pdb")):
            continue
        pep, bl = d.name.split("__")
        if pep in seqs:
            jobs.append((d.name, bl[:-1], seqs[pep]))
    print(f"{len(jobs)} cells to score with our dG model ({len(done)} done), "
          f"{workers} workers", flush=True)
    if not jobs:
        return

    with OUT.open("a") as fh, ProcessPoolExecutor(workers, initializer=_init) as ex:
        for k, res in enumerate(ex.map(score_cell, jobs, chunksize=1), 1):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            if k % 20 == 0 or k == 1:
                v = res["best_iface"]
                print(f"  [{k}/{len(jobs)}] {res['name']:20s} "
                      f"dG {'--' if v is None else f'{v:7.2f}'} kcal/mol "
                      f"({res['n_scored']}/24 poses)", flush=True)
    print("COVENTRY_DG_DONE")


if __name__ == "__main__":
    main()
