"""E249 — is CHARGED the real issue, or is our pipeline/CV sabotaging us? The control: run the IDENTICAL
features + IDENTICAL clustered-CV on HYDROPHOBIC vs CHARGED vs ALL SKEMPI mutations. If the pipeline were
broken (bad CV, bad labels, bad features), hydrophobic would fail too. If hydrophobic works and charged
doesn't on the SAME code, it's physics, not a bug. Also: sign sanity-check + within/between decomposition,
and the deeper reframe — is the wall 'charged' or is it the cross-target OFFSET (receptor-baseline)?
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold, KFold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def R(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); m = ~(np.isnan(a) | np.isnan(b))
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 3 else np.nan


def main():
    rows = [json.loads(l) for l in open(ROOT / "data/e165_skempi_struct.jsonl")]
    print(f"=== SKEMPI total n={len(rows)} ===")

    # sign sanity: largest-|ddg| mutations — destabilizing alanine scans should be POSITIVE ddg
    sd = sorted(rows, key=lambda r: -r["ddg"])[:3] + sorted(rows, key=lambda r: r["ddg"])[:3]
    print("  sign check (ddg>0 should = destabilizing):")
    for r in sd:
        print(f"    {r['key']:<22} wt={r['wt']}->{r['mutaa']} ddg={r['ddg']:+.2f} burial={r.get('burial')}")

    def feats(r):
        return [float(r.get(k, 0) or 0) for k in ("burial", "iface_dist", "n5", "n8")]

    def cv(sub, splitter, grp=None):
        y = np.array([r["ddg"] for r in sub]); X = np.nan_to_num([feats(r) for r in sub])
        pred = np.full(len(sub), np.nan)
        g = np.array([r["pdb"] for r in sub])
        sp = splitter.split(X, y, g) if grp else splitter.split(X, y)
        for tr, te in sp:
            pred[te] = HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=0.05,
                                                     random_state=0).fit(X[tr], y[tr]).predict(X[te])
        return R(pred, y), y, g, pred

    subsets = {"HYDROPHOBIC (wt∈AILMFWVY)": [r for r in rows if r["wt"] in "AILMFWVY"],
               "CHARGED (wt∈DEKR)": [r for r in rows if r["wt"] in "DEKR"],
               "POLAR (wt∈STNQHY)": [r for r in rows if r["wt"] in "STNQHY"],
               "ALL": rows}
    print(f"\n  {'subset':<26}{'n':>6}{'POOLED-CV':>11}{'CLUSTERED-CV':>13}{'within-cent':>12}  (SAME pipeline)")
    for nm, sub in subsets.items():
        if len(sub) < 50:
            continue
        npk = len(set(r["pdb"] for r in sub))
        pooled, *_ = cv(sub, KFold(5, shuffle=True, random_state=0))
        clust, y, g, pred = cv(sub, GroupKFold(min(8, npk)), grp=True)
        pm = {p: y[g == p].mean() for p in set(g)}
        yc = y - np.array([pm[p] for p in g]); pc = pred - np.array([pred[g == p].mean() for p in g])
        print(f"  {nm:<26}{len(sub):>6}{pooled:>+11.3f}{clust:>+13.3f}{R(pc, yc):>+12.3f}")

    # the reframe: is the OFFSET (per-pocket mean ddg) the thing that fails — for EVERY residue class?
    print("\n  === is the cross-pocket OFFSET the universal wall (not 'charged')? ===")
    print(f"  {'subset':<26}{'between%':>9}{'offset LOO r':>14}")
    for nm, sub in subsets.items():
        if len(sub) < 50:
            continue
        g = np.array([r["pdb"] for r in sub]); y = np.array([r["ddg"] for r in sub])
        pk = sorted(set(g)); pm = np.array([y[g == p].mean() for p in pk])
        tot = y.var(); btw = pm.var()
        # can pocket-mean burial predict pocket-mean ddg (the offset) out-of-sample?
        X = np.array([[np.mean([float(r.get("burial", 0) or 0) for r in sub if r["pdb"] == p]),
                       np.mean([float(r.get("n8", 0) or 0) for r in sub if r["pdb"] == p])] for p in pk])
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        po = np.full(len(pk), np.nan)
        for i in range(len(pk)):
            tr = [j for j in range(len(pk)) if j != i]
            if len(tr) < 5:
                continue
            sc = StandardScaler().fit(X[tr]); po[i] = Ridge(alpha=2.0).fit(sc.transform(X[tr]), pm[tr]).predict(sc.transform(X[i:i+1]))[0]
        print(f"  {nm:<26}{100*btw/tot:>8.0f}%{R(po, pm):>+14.3f}")


if __name__ == "__main__":
    main()
