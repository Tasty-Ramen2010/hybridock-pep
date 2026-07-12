"""E129 — cross-dataset sign-stability of anchor features (the gate before wiring to calibration).

Anchor features were validated on PDBbind crystal poses. The discipline: a feature must hold its sign on
ANOTHER dataset or it's unreliable. Compute anchor features on the98 (e95 real RAPiDock poses + receptors)
and check corr(feature, ΔG) — same sign as PDBbind? Also sanity-checks the module runs on production
PDB inputs (peptide pose PDB + receptor PDB).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hybridock_pep.scoring.anchor_features import ANCHOR_FEATURE_KEYS, compute_anchor_features  # noqa: E402

E95 = ROOT / "runs" / "e95_the98_campaign"
# PDBbind reference signs (E128, corr with ΔG on short; broadly: more buried/anchored → stronger → ΔG down)
PDBBIND_SIGN = {"max_burial": -1, "burial_concentration": +1, "best_salt_bridge": -1,
                "charged_anchor": -1, "buried_inert": -1, "pro_run": +1}


def load_the98_y():
    y = {}
    for fn in ["data/pooled_benchmark_train.csv", "data/pooled_benchmark_test.csv"]:
        for r in csv.DictReader(open(ROOT / fn)):
            if r["dataset"] == "the98":
                y[r["pdb"]] = float(r["y"])
    return y


def main():
    ydict = load_the98_y()
    rows = []
    for cid, yval in ydict.items():
        d = E95 / cid
        rec = d / "receptor.pdb"
        pose = d / "poses" / "pose_0.pdb"
        if not (rec.exists() and pose.exists()):
            continue
        af = compute_anchor_features(pose, rec)
        if af:
            rows.append((yval, af))
    print(f"=== E129 anchor cross-dataset check on the98 (n={len(rows)}, rank-1 real poses) ===\n")
    if len(rows) < 15:
        print("  too few the98 structures found.")
        return
    y = np.array([r[0] for r in rows])
    print(f"  {'feature':<22}{'the98 corr':>12}{'PDBbind sign':>14}{'  STABLE?':>10}")
    nstable = 0
    for k in ANCHOR_FEATURE_KEYS:
        v = np.array([r[1][k] for r in rows])
        m = ~np.isnan(v)
        r = pearsonr(v[m], y[m])[0] if m.sum() > 4 and np.std(v[m]) > 0 else np.nan
        stable = (np.sign(r) == PDBBIND_SIGN[k]) and abs(r) > 0.05
        nstable += stable
        print(f"  {k:<22}{r:>+12.3f}{'(' + ('-' if PDBBIND_SIGN[k] < 0 else '+') + ')':>14}{'OK' if stable else 'flip/weak':>10}")
    print(f"\n  → {nstable}/{len(ANCHOR_FEATURE_KEYS)} anchor features sign-stable PDBbind→the98.")
    print("  ≥4 stable ⇒ wire to the pooled calibration; flips ⇒ keep only the sign-stable ones.")


if __name__ == "__main__":
    main()
