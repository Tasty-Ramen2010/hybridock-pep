"""E18v2 MD-entropy eval — finally test Ram's 100ps Ramachandran entropy hypothesis.

Uses /tmp/e18v2_cr.json (ds_dih = Σ(S_bound-S_free) dihedral-histogram entropy; rmsf_ratio
= free/bound RMSF rigidification) computed on all 65 crystal complexes by e18v2_features.py.

Honest LOO tests:
  - does ds_dih correlate with ΔG alone? rmsf_ratio alone?
  - does adding them to the pocket+interface geometry model IMPROVE r / RMSE?
Verdict: is the 100ps structure-based entropy worth its ~500x cost over instant geometry?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

POCK = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis"]
IFACE = ["bsa_hyd", "sasa_hb", "sasa_sb", "arom_cc", "hb_count"]


def _loo_pred(rows, feats, y):
    X = np.array([[r.get(f, 0.0) for f in feats] for r in rows], float)
    pred = np.zeros(len(y))
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd])
        w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pred[i] = np.r_[1, (X[i] - mu) / sd] @ w
    return pred


def rr(rows, feats, y):
    p = _loo_pred(rows, feats, y)
    return pearsonr(p, y).statistic, float(np.sqrt(np.mean((p - y) ** 2)))


def main():
    md = json.loads(Path("/tmp/e18v2_cr.json").read_text())
    # join geometry features (e19_cr) by pdb
    geo = {r["pdb"].upper(): r for r in json.loads(Path("/tmp/e19_cr.json").read_text())}
    rows, y = [], []
    for m in md:
        g = geo.get(m["pdb"].upper())
        if not g:
            continue
        rec = {**g, "ds_dih": m["ds_dih"], "rmsf_ratio": m["rmsf_ratio"],
               "de_strength": m.get("de_strength", 0.0)}
        rows.append(rec); y.append(m["y"])
    y = np.array(y)
    print(f"n={len(rows)} complexes with both MD-entropy and geometry features\n")

    print("=== raw correlation of MD-entropy terms with ΔG ===")
    for f in ["ds_dih", "rmsf_ratio", "de_strength"]:
        v = np.array([r[f] for r in rows])
        if v.std() > 0:
            print(f"  corr({f:<12}, ΔG) = {pearsonr(v, y).statistic:+.3f}")

    print("\n=== LOO: does MD-entropy ADD to pocket+interface geometry? ===")
    print(f"{'model':<34}{'r':>8}{'RMSE':>8}")
    combos = {
        "geometry (pocket+interface)": POCK + IFACE,
        "  + ds_dih": POCK + IFACE + ["ds_dih"],
        "  + rmsf_ratio": POCK + IFACE + ["rmsf_ratio"],
        "  + both MD-entropy": POCK + IFACE + ["ds_dih", "rmsf_ratio"],
        "MD-entropy ALONE (ds+rmsf)": ["ds_dih", "rmsf_ratio"],
    }
    base_r = None
    for name, fs in combos.items():
        r, rmse = rr(rows, fs, y)
        if "geometry (pocket" in name:
            base_r = r
        print(f"{name:<34}{r:>8.3f}{rmse:>8.2f}")
    print(f"\nguess-the-mean RMSE = {y.std():.2f}")
    print(f"VERDICT: MD-entropy {'HELPS' if base_r is not None else ''} "
          "— compare '+both' r to 'geometry' r; if not >+0.03, the 100ps cost is not justified.")


if __name__ == "__main__":
    main()
