#!/usr/bin/env python
"""Our features on GOOD poses — the missing cell of the 2x2 that says what we are actually short of.

THE 2x2. Selectivity on this grid can fail for two reasons and they need separating, because they
imply completely different work:

                          ref2015 total        our 18 features
      our docked poses    AUC 0.735            AUC 0.628   (learned, trained on docked blocks)
      co-folded poses     AUC 0.820            ???  <-- this script

Three of the four are measured. The fourth decides the diagnosis:

  if our features reach ~0.8 on co-folded poses   the feature set is FINE and every point of the
                                                  deficit is pose generation. The work is a better
                                                  generator, and our scoring stays as it is.
  if they stay near chance on co-folded poses     the feature set is ALSO wrong, and a better
                                                  generator alone would not save us -- the
                                                  descriptors do not encode what distinguishes a
                                                  cognate pair from a near-miss.

Nothing is refitted here: the model trained on docked blocks is applied unchanged, so this is a
transfer measurement and not a new fit. The features are extracted with exactly the extractor the
Coventry docked cells used -- same hard repack, no soft-repulsive pre-pass -- so the only thing
that differs between the two rows of the table is which structure was scored.

The four binders whose ESMFold model is the proven outlier are read from their AF3 re-transplant,
because the peptide for those cells lives in the AF3 frame and scoring it against our ESMFold
model returns interface energies in the thousands of REU.

Usage: sel_cofold_features.py [workers]
Output: logs/sel_cofold_features.jsonl
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "scripts"))
BOLTZ = ROOT / "runs/coventry/boltz"
BOLTZ_AF3 = ROOT / "runs/coventry/boltz_af3"
BINDERS = ROOT / "datasets/coventry/binders"
AF3_BINDERS = ROOT / "datasets/coventry/binders_af3"
OUT = ROOT / "logs/sel_cofold_features.jsonl"

ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
BROKEN = {"n3", "n7", "pc21", "pc26"}

_PR = None
_SF = None


def _init() -> None:
    global _PR, _SF
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ["SEL_SOFT_PASS"] = "0"      # a co-folded pose's clashes are real, as with docked
    sys.path.insert(0, str(ROOT / "scripts"))
    from coventry_refine import init_rosetta
    _PR = init_rosetta()
    _SF = _PR.create_score_function("ref2015")
    import sel_features
    sel_features._PR = _PR
    sel_features._SF = _SF
    import sel_coventry_features as SCF
    SCF._PR = _PR
    SCF._SF = _SF
    SCF.SOFT_PASS = False


def do_cell(args: tuple) -> dict:
    import sel_coventry_features as SCF
    pep, binder = args
    name = f"{pep}__{binder}"
    row = {"name": name, "peptide": pep, "binder": binder, "cognate": int(pep == binder)}
    af3 = binder in BROKEN
    pose = (BOLTZ_AF3 if af3 else BOLTZ) / name / "peptide.pdb"
    if not pose.exists():
        pose = BOLTZ / name / "peptide.pdb"
        af3 = False
    receptor = (AF3_BINDERS if af3 else BINDERS) / f"{binder}_1b1.pdb"
    row["receptor_model"] = "af3" if af3 else "esmfold"
    if not pose.exists():
        row["error"] = "no transplanted peptide"
        return row
    t0 = time.time()
    try:
        row["best"] = {k: float(v) for k, v in
                       SCF.refine_and_feature(str(receptor), str(pose)).items()
                       if isinstance(v, (int, float))}
    except Exception as exc:  # noqa: BLE001 -- one bad cell must not stop the grid
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = round(time.time() - t0, 1)
    return row


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 10
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["name"])
    jobs = [(p, b) for p in ORDER for b in ORDER if f"{p}__{b}" not in done]
    print(f"{len(jobs)} co-folded cells to featurise, {workers} workers", flush=True)
    if not jobs:
        return
    t0 = time.time()
    with OUT.open("a") as fh, ProcessPoolExecutor(workers, initializer=_init) as ex:
        for k, row in enumerate(ex.map(do_cell, jobs, chunksize=1), 1):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if k % 25 == 0 or k == 1:
                v = row.get("best", {}).get("i_total", float("nan"))
                el = time.time() - t0
                print(f"  [{k}/{len(jobs)}] {row['name']:14s} i_total {v:8.1f}  "
                      f"eta {(len(jobs) - k) * el / k / 60:.0f} min", flush=True)
    print("SEL_COFOLD_FEATURES_DONE")


if __name__ == "__main__":
    main()
