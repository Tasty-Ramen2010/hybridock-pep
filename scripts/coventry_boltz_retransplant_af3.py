#!/usr/bin/env python
"""Rescue the fold-gate rejects by transplanting onto the AF3 binder model instead.

Every fold-gate reject so far is in one of four columns -- n3, n7, pc21, pc26 -- and those are
exactly the four binders where our ESMFold model is the proven outlier: Boltz and AlphaFold 3,
two independent predictors given the identical paper sequence, agree with each other to
1.3/2.4/1.6/0.9 A and both disagree with our model by 16.9/14.6/8.7/5.7 A.

So the gate is not misfiring. It is correctly reporting that the co-folded binder and the model
we docked against are different structures -- and the one that is wrong is ours. Retrying the
co-fold would not help, because Boltz is already right. The fix is to transplant onto the AF3
model, which is free: the co-folded structures are already on disk and the transplant is pure
geometry, no GPU.

This runs on the main grid's existing output and writes a parallel record, so the original
ESMFold-transplanted numbers stay on disk for comparison.

Usage: coventry_boltz_retransplant_af3.py
Output: logs/coventry_boltz_af3.jsonl (fed to the same scorer via COVENTRY_BOLTZ_SRC)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "src"))

MAIN = ROOT / "logs/coventry_boltz.jsonl"
OUT = ROOT / "logs/coventry_boltz_af3.jsonl"
AF3 = ROOT / "datasets/coventry/binders_af3"
POSES = ROOT / "runs/coventry/boltz_af3"
#: binders where two independent predictors agree against our ESMFold model
BROKEN = {"n3", "n7", "pc21", "pc26"}


def main() -> None:
    from hybridock_pep.analysis.direction import axis_quality
    from hybridock_pep.sampling.cofold import (transplant, write_transplanted_peptide,
                                               write_transplanted_pose)

    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["name"])

    n_ok = n_still = 0
    with OUT.open("a") as fh:
        srcs = [f for f in sorted(ROOT.glob("logs/coventry_boltz*.jsonl"))
                if all(k not in f.name for k in ("refine", "retry", "af3"))]
        for line in (l for f in srcs for l in f.read_text().splitlines()):
            if not line.strip():
                continue
            m = json.loads(line)
            if m["binder"] not in BROKEN or m["name"] in done or "pdb" not in m:
                continue
            rec_pdb = AF3 / f"{m['binder']}_1b1.pdb"
            if not rec_pdb.exists():
                continue
            out = dict(m)
            out["receptor_model"] = "af3"
            out["fold_rmsd_esmfold"] = m.get("fold_rmsd")
            try:
                tr = transplant(Path(m["pdb"]), rec_pdb)
            except ValueError as exc:
                out["transplant_error"] = str(exc)
                fh.write(json.dumps(out) + "\n")
                continue
            d = POSES / m["name"]
            write_transplanted_pose(tr, rec_pdb, d / "transplanted.pdb")
            out.update({
                "fold_rmsd": round(tr.fold_rmsd, 3),
                "n_aligned": tr.n_aligned,
                "accepted": tr.accepted,
                "axis": [round(float(v), 4) for v in tr.direction],
                "axis_quality": round(axis_quality(tr.peptide_ca), 3),
                "transplanted": str(d / "transplanted.pdb"),
                "peptide_only": str(write_transplanted_peptide(tr, d / "peptide.pdb")),
            })
            fh.write(json.dumps(out) + "\n")
            n_ok += tr.accepted
            n_still += not tr.accepted
            print(f"  {m['name']:18s} ESMFold {out['fold_rmsd_esmfold']:6.2f} A -> "
                  f"AF3 {tr.fold_rmsd:5.2f} A   "
                  f"{'RESCUED' if tr.accepted else 'still rejected'}", flush=True)
    print(f"\n{n_ok} rescued, {n_still} still over the gate")


if __name__ == "__main__":
    main()
