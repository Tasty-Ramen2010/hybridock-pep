"""E219 — Ram's charged idea: replace hand-crafted charge indicators with RICHER charge-interaction features
+ structure (salt-bridge burial), let the model learn strong vs weak charged bonds from the full dataset.

Current charge features = 3 net-charge-complementarity scalars. New: per-charge-pair interaction profile
(favorable +/- pairs, unfavorable like-like, peptide charge × pocket charge matrix), charge×burial (a buried
salt bridge is strong, a surface one is screened/weak), charge×pocket-hydrophobicity (charge in a hydrophobic
pocket = strong desolvation penalty). Compare charged r: base vs +charge-interaction, crystal-925 clustered-CV.
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
from hybridock_pep.scoring.affinity_model import build_feature_vector, GEOMETRY_KEYS, _SCALES  # noqa: E402
import e158_overfit_failure_analysis as e158  # noqa: E402
import e202_band_routing_build as e202  # noqa: E402
SN = list(_SCALES.keys())


def charge_interaction(seq, ps, geom):
    """richer charge-interaction features (structure-free + burial-modulated)."""
    pp = sum(c == "K" or c == "R" for c in seq); pn = sum(c == "D" or c == "E" for c in seq)
    php = sum(c == "H" for c in seq)
    op = sum(c in "KR" for c in ps); on = sum(c in "DE" for c in ps); oh = sum(c == "H" for c in ps)
    npoc = max(len(ps), 1)
    burial = float(geom.get("mean_burial", 0.0)); pock_hyd = float(np.mean([_SCALES["kd"].get(c, 0) for c in ps])) if ps else 0.0
    f_op = (op + 0.5 * oh) / npoc; f_on = on / npoc  # pocket +/- fractions
    # favorable pairs (pep+ with pocket-, pep- with pocket+); unfavorable like-like
    favorable = pp * f_on + pn * f_op
    unfavorable = pp * f_op + pn * f_on
    netbal = favorable - unfavorable
    return [
        float(pp), float(pn), float(php), float(pp - pn),           # peptide charge profile
        f_op, f_on,                                                 # pocket charge profile
        favorable, unfavorable, netbal,                             # pairing
        netbal * burial / 100.0,                                    # buried salt bridge = STRONG
        (pp + pn) * pock_hyd,                                       # charge in hydrophobic pocket = desolv penalty
        netbal * (1.0 - min(abs(pock_hyd) / 4.0, 1.0)),            # charge in polar pocket = screened/weaker
        abs(pp - pn) * burial / 100.0,                             # net charge × burial
    ]


def main():
    rows = []
    for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")):
        ps = e158.pocket_seq(r["pdb"])
        if ps is None:
            continue
        g = {k: float(r.get(k, 0.0)) for k in GEOMETRY_KEYS}; g["pocket_seq"] = ps
        x = build_feature_vector(g, r["seq"])
        base = (x[:262] if x.shape[0] >= 262 else np.pad(x, (0, 262 - x.shape[0]))).tolist()
        ci = charge_interaction(r["seq"], ps, g)
        rows.append({"base": base, "ci": ci, "y": float(r["y"]), "L": r["length"], "ps": ps,
                     "q": abs(sum(c in "KR" for c in r["seq"]) - sum(c in "DE" for c in r["seq"]))})
    y = np.array([r["y"] for r in rows]); q = np.array([r["q"] for r in rows]); L = np.array([r["L"] for r in rows])
    grp, _ = e158.greedy_cluster([r["ps"] for r in rows], 0.7)

    def cv(withci):
        X = np.nan_to_num([r["base"] + (r["ci"] if withci else []) for r in rows])
        pred = np.full(len(rows), np.nan)
        for tr, te in GroupKFold(5).split(X, y, grp):
            pred[te] = e202._hgb().fit(X[tr], y[tr]).predict(X[te])
        return pred

    def R(p, m):
        ok = ~np.isnan(p[m]); return float(np.corrcoef(p[m][ok], y[m][ok])[0, 1]) if ok.sum() > 4 else float("nan")
    pb, pc = cv(False), cv(True)
    print("=== Ram's charge-interaction features (vs current net-charge-complementarity) ===")
    print(f"  {'slice':<16}{'n':>5}{'base r':>9}{'+charge-int':>13}{'Δ':>8}")
    for nm, m in [("OVERALL", np.ones(len(rows), bool)), ("charged|q|>=2", q >= 2), ("|q|>=3", q >= 3),
                  ("charged ≤12", (q >= 2) & (L <= 12)), ("neutral|q|<=1", q <= 1)]:
        print(f"  {nm:<16}{int(m.sum()):>5}{R(pb,m):>+9.3f}{R(pc,m):>+13.3f}{R(pc,m)-R(pb,m):>+8.3f}")
    print("\n  (memory: charged floor = single-pose electrostatics/desolvation, FEP-bound — test if richer")
    print("   structure-modulated charge interaction breaks it or confirms the floor)")


if __name__ == "__main__":
    main()
