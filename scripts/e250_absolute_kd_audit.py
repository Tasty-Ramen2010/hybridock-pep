"""E250 — the sabotage control for ABSOLUTE Kd (the real product metric, not ΔΔG). On the 925 PDBbind peptide
complexes: run the IDENTICAL production-feature pipeline + IDENTICAL clustered-CV, split by CHARGED vs NEUTRAL
peptide. If neutral predicts well and charged collapses on the SAME code → physics, not a bug. Also: is the
charged failure a FEATURE problem (do charged complexes just need electrostatics) or a LABEL/transfer problem?
Decompose: pooled-CV vs clustered-CV (novel receptor) vs feature-by-feature charged-vs-neutral correlation.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold, KFold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e158_overfit_failure_analysis as e158  # noqa: E402
FEATS = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
         "arom_cc", "hb_count", "mj_contact", "strength_bur", "rg_per_L", "org_density", "cys_frac", "mean_burial"]


def R(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); m = ~(np.isnan(a) | np.isnan(b))
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 3 else np.nan


def netq(seq):
    return sum(c in "KR" for c in seq) - sum(c in "DE" for c in seq)


def main():
    rows = [json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")]
    for r in rows:
        r["q"] = abs(netq(r["seq"]))
    print(f"=== ABSOLUTE Kd sabotage control: n={len(rows)} PDBbind peptide complexes ===")

    # cluster by pocket-seq for honest no-leak clustered CV
    pseqs = []
    keep = []
    for r in rows:
        ps = e158.pocket_seq(r["pdb"])
        if ps:
            r["ps"] = ps; keep.append(r); pseqs.append(ps)
    rows = keep
    grp, _ = e158.greedy_cluster([r["ps"] for r in rows], 0.7)
    print(f"  usable n={len(rows)}, homology clusters={len(set(grp))}")

    def feat(r):
        return [float(r.get(k, 0) or 0) for k in FEATS]

    def cv(sub, gsub, splitter, grouped):
        y = np.array([r["y"] for r in sub]); X = np.nan_to_num([feat(r) for r in sub])
        pred = np.full(len(sub), np.nan)
        sp = splitter.split(X, y, gsub) if grouped else splitter.split(X, y)
        for tr, te in sp:
            pred[te] = HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.04,
                                                     l2_regularization=3.0, random_state=0).fit(X[tr], y[tr]).predict(X[te])
        return R(pred, y)

    grp = np.array(grp)
    subsets = {
        "NEUTRAL |q|<=1": [(r, g) for r, g in zip(rows, grp) if r["q"] <= 1],
        "CHARGED |q|>=2": [(r, g) for r, g in zip(rows, grp) if r["q"] >= 2],
        "V.CHARGED |q|>=3": [(r, g) for r, g in zip(rows, grp) if r["q"] >= 3],
        "ALL": list(zip(rows, grp)),
    }
    print(f"\n  {'subset':<20}{'n':>6}{'POOLED-CV':>11}{'CLUSTERED-CV':>14}  (IDENTICAL pipeline)")
    for nm, sg in subsets.items():
        if len(sg) < 40:
            continue
        sub = [x[0] for x in sg]; gsub = np.array([x[1] for x in sg])
        pooled = cv(sub, gsub, KFold(5, shuffle=True, random_state=0), False)
        clust = cv(sub, gsub, GroupKFold(min(5, len(set(gsub)))), True)
        print(f"  {nm:<20}{len(sub):>6}{pooled:>+11.3f}{clust:>+14.3f}")

    # the key question: is charged a FEATURE gap (electrostatics missing) or a LABEL-NOISE/spread gap?
    print("\n  === diagnosis: WHY do charged complexes fail (label spread + feature signal) ===")
    for nm, sg in subsets.items():
        if len(sg) < 40:
            continue
        sub = [x[0] for x in sg]; y = np.array([r["y"] for r in sub])
        # best single structural feature
        best = max(FEATS, key=lambda k: abs(R([float(r.get(k, 0) or 0) for r in sub], y)) if not np.isnan(R([float(r.get(k, 0) or 0) for r in sub], y)) else 0)
        br = R([float(r.get(best, 0) or 0) for r in sub], y)
        print(f"  {nm:<20} label std={y.std():.2f}  best feat={best}({br:+.2f})  "
              f"mj_contact r={R([float(r.get('mj_contact', 0) or 0) for r in sub], y):+.2f}  "
              f"bsa_hyd r={R([float(r.get('bsa_hyd', 0) or 0) for r in sub], y):+.2f}")


if __name__ == "__main__":
    main()
