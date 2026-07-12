"""E226 — does fpocket apo-pocket physics break the wall? Per-complex ΔG clustered-CV (novel receptor),
baseline (peptide-ProtDCal + our pocket composition) vs +fpocket-physics (18 druggability/water descriptors).
Also: do fpocket descriptors predict the receptor baseline better than sequence (0.15)?
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from hybridock_pep.scoring.affinity_model import _protdcal_descriptors, _SCALES  # noqa: E402
import e158_overfit_failure_analysis as e158  # noqa: E402
import e202_band_routing_build as e202  # noqa: E402
SN = list(_SCALES.keys())


def R(p, y, m=None):
    if m is not None:
        p, y = p[m], y[m]
    ok = ~(np.isnan(p) | np.isnan(y))
    return float(np.corrcoef(p[ok], y[ok])[0, 1]) if ok.sum() > 4 else float("nan")


def main():
    fp = {json.loads(l)["pdb"].lower(): json.loads(l)["fp"]
          for l in open(ROOT / "data/e225_fpocket.jsonl") if json.loads(l).get("fp")}
    rows = []
    for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")):
        pid = r["pdb"].lower()
        if pid not in fp:
            continue
        ps = e158.pocket_seq(pid)
        if ps is None:
            continue
        pkf = [float(np.mean([_SCALES[s].get(c, 0) for c in ps])) for s in SN]
        rows.append({"seq": r["seq"], "y": float(r["y"]), "ps": ps, "pkf": pkf, "fp": fp[pid], "L": r["length"]})
    print(f"=== fpocket-physics eval: n={len(rows)} (have fpocket descriptors) ===", flush=True)
    y = np.array([r["y"] for r in rows])
    grp, _ = e158.greedy_cluster([r["ps"] for r in rows], 0.7)

    def feat(r, withfp):
        f = _protdcal_descriptors(r["seq"]) + r["pkf"] + [float(r["L"])]
        return f + (r["fp"] if withfp else [])

    def cv(withfp):
        X = np.nan_to_num([feat(r, withfp) for r in rows]); pred = np.full(len(rows), np.nan)
        for tr, te in GroupKFold(5).split(X, y, grp):
            pred[te] = e202._hgb().fit(X[tr], y[tr]).predict(X[te])
        return pred

    pb, pf = cv(False), cv(True)
    print("\n=== per-complex ΔG (clustered-CV, NOVEL receptor) ===")
    print(f"  baseline (pep+pocket-comp):     r={R(pb, y):+.3f}")
    print(f"  + fpocket apo-pocket physics:   r={R(pf, y):+.3f}   Δ={R(pf,y)-R(pb,y):+.3f}")

    # fpocket-only: how much does pocket physics ALONE explain affinity?
    Xfp = np.nan_to_num([r["fp"] for r in rows]); po = np.full(len(rows), np.nan)
    for tr, te in GroupKFold(5).split(Xfp, y, grp):
        po[te] = e202._hgb().fit(Xfp[tr], y[tr]).predict(Xfp[te])
    print(f"  fpocket-physics ALONE:          r={R(po, y):+.3f}  (pure pocket-bindability → affinity)")

    # which fpocket descriptors correlate with affinity?
    DKEYS = ["Score", "Druggability", "nAlphaSph", "TotalSASA", "PolarSASA", "ApolarSASA", "Volume",
             "HydrophobDensity", "AlphaSphRadius", "SolvAccess", "ApolarProp", "Hydrophobicity", "VolumeScore",
             "Polarity", "Charge", "PolarAtomProp", "AlphaSphDensity", "MaxDist", "Flexibility"]
    print("\n  fpocket descriptors vs affinity (|r|>0.10):")
    for i, k in enumerate(DKEYS):
        c = np.corrcoef(Xfp[:, i], y)[0, 1]
        if abs(c) > 0.10:
            print(f"    {k:<18} r={c:+.3f}")


if __name__ == "__main__":
    main()
