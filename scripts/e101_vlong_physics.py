"""E101 — is vlong 'failing physics', or is the physics fine and something else is missing?

Ram's principle: physics doesn't lie. If vlong (≥17) scores badly, the enthalpic physics isn't wrong —
either the LABELS have no range (can't correlate) or a real physical TERM is missing (conformational
entropy, which scales with length and is invisible to a static pose). Tested on COMBINED crystal (n=156,
pooled benchmark CSVs — the documented 0.544 set, NOT the degenerate cr65-only slice).

1. LABEL RANGE per band per dataset — is the 'failure' just no affinity variance to predict?
2. IS THE ENTHALPY REAL? within long+vlong (≥13), do enthalpic features (contacts/burial/H-bond/salt)
   correlate with y? If yes → physics works for long peptides given range.
3. MISSING-TERM SIGNATURE — LOO residual vs length / org_density / rg_per_L. If we systematically
   OVER-predict affinity for long FLEXIBLE peptides (residual = y−pred > 0, ∝ length×disorder), that is
   the missing conformational-entropy penalty, not the enthalpy being wrong.
4. ENTROPY-TERM TEST — add an explicit length / (1−org) penalty; does long/vlong r improve?
"""
from __future__ import annotations

import csv
import os
import sys
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]

import numpy as np  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

PROD = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
        "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density", "cys_frac"]
ENTHALPIC = ["mj_contact", "bsa_hyd", "sasa_hb", "sasa_sb", "poc_n", "mean_burial", "hb_count", "arom_cc"]


def band(L):
    return "short≤8" if L <= 8 else "med9-12" if L <= 12 else "long13-16" if L <= 16 else "vlong≥17"


def load():
    rows = []
    for f in ["data/pooled_benchmark_train.csv", "data/pooled_benchmark_test.csv"]:
        for r in csv.DictReader(open(ROOT / f)):
            rows.append({"y": float(r["y"]), "length": int(float(r["length"])), "dataset": r["dataset"],
                         "net_charge": float(r["net_charge"]), "seq": r.get("seq", ""),
                         "feat": {c: float(r[c]) for c in PROD}})
    return rows


def ridge(rows, cols, lam=1.0):
    X = np.array([[r["feat"][c] for c in cols] for r in rows], float)
    y = np.array([r["y"] for r in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    R = np.eye(A.shape[1]) * lam
    R[0, 0] = 0
    return mu, sd, np.linalg.solve(A.T @ A + R, A.T @ y)


def predict(feat, cols, p):
    mu, sd, w = p
    x = np.array([feat[c] for c in cols], float)
    return float(np.r_[1.0, (x - mu) / sd] @ w)


def loo(rows, cols):
    pred = np.full(len(rows), np.nan)
    for i in range(len(rows)):
        tr = [rows[j] for j in range(len(rows)) if j != i]
        pred[i] = predict(rows[i]["feat"], cols, ridge(tr, cols))
    return pred


def rstat(p, y):
    m = ~(np.isnan(p) | np.isnan(y))
    return (pearsonr(p[m], y[m])[0], float(np.sqrt(np.mean((p[m] - y[m]) ** 2))), int(m.sum())) if m.sum() > 4 else (np.nan, np.nan, int(m.sum()))


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    return pearsonr(x[m], y[m])[0] if m.sum() > 4 and np.std(x[m]) > 0 else np.nan


def main():
    rows = load()
    y = np.array([r["y"] for r in rows])
    L = np.array([r["length"] for r in rows])
    print(f"=== E101 vlong PHYSICS on COMBINED crystal (n={len(rows)}) ===\n")

    print("1. LABEL RANGE per band per dataset (is there variance to predict?):")
    for ds in ["cr65", "the98", "ALL"]:
        for b in ["med9-12", "long13-16", "vlong≥17"]:
            sub = [r for r in rows if band(r["length"]) == b and (ds == "ALL" or r["dataset"] == ds)]
            if len(sub) >= 3:
                yy = np.array([r["y"] for r in sub])
                print(f"   {ds:<5} {b:<11} n={len(sub):<3} y-range={yy.max()-yy.min():.1f} kcal  std={yy.std():.2f}")
        print()

    print("2. IS THE ENTHALPY REAL? within long+vlong (≥13, combined) feature→y correlations:")
    lv = [r for r in rows if r["length"] >= 13]
    ylv = np.array([r["y"] for r in lv])
    for c in ENTHALPIC + ["strength_bur", "rg_per_L", "org_density", "cys_frac"]:
        rr = corr([r["feat"][c] for r in lv], ylv)
        tag = "  enthalpic" if c in ENTHALPIC else ""
        print(f"   {c:<14} r(feat,y)={rr:+.3f}{tag}")
    # enthalpy-only model on long+vlong
    pe = loo(lv, ENTHALPIC)
    print(f"   → enthalpy-only LOO on ≥13:  r={rstat(pe, ylv)[0]:+.3f}  (physics works IF >0)")

    print("\n3. MISSING-TERM SIGNATURE — residual (y−pred) vs length & flexibility (combined LOO):")
    pred = loo(rows, PROD)
    resid = y - pred
    print(f"   corr(residual, length)        = {corr(L, resid):+.3f}")
    print(f"   corr(residual, org_density)   = {corr([r['feat']['org_density'] for r in rows], resid):+.3f}")
    print(f"   corr(residual, rg_per_L)      = {corr([r['feat']['rg_per_L'] for r in rows], resid):+.3f}")
    print(f"   corr(residual, |net_charge|)  = {corr([abs(r['net_charge']) for r in rows], resid):+.3f}")
    lvm = L >= 13
    print(f"   within ≥13:  corr(resid,length)={corr(L[lvm], resid[lvm]):+.3f}  "
          f"corr(resid,org_density)={corr([r['feat']['org_density'] for r in rows if r['length']>=13], resid[lvm]):+.3f}")
    over = resid[lvm] > 0
    print(f"   ≥13 mean residual={np.nanmean(resid[lvm]):+.2f} kcal  (>0 = we OVER-predict binding strength)  "
          f"frac over-predicted={over.mean():.2f}")

    print("\n4. ENTROPY-TERM TEST — add length-scaled disorder penalty (combined LOO, per band):")
    # entropy proxy: length * (1 - org_density_normalized) — flexible long peptides pay more
    od = np.array([r["feat"]["org_density"] for r in rows])
    odn = (od - od.min()) / (od.max() - od.min() + 1e-9)
    for r, e in zip(rows, L * (1 - odn)):
        r["feat"]["ent_pen"] = float(e)
    for r, e in zip(rows, L.astype(float)):
        r["feat"]["len_raw"] = float(e)
    base = loo(rows, PROD)
    plus_ent = loo(rows, PROD + ["ent_pen"])
    plus_len = loo(rows, PROD + ["len_raw"])
    for nm, p in [("PROD (base)", base), ("PROD + len*(1-org) entropy", plus_ent), ("PROD + raw length", plus_len)]:
        s = rstat(p, y)
        bands_r = {b: rstat(p[np.array([band(x) == b for x in L])], y[np.array([band(x) == b for x in L])])[0]
                   for b in ["med9-12", "long13-16", "vlong≥17"]}
        print(f"   {nm:<28} pooled r={s[0]:+.3f} | med {bands_r['med9-12']:+.2f} long {bands_r['long13-16']:+.2f} vlong {bands_r['vlong≥17']:+.2f}")

    print("\n  VERDICT: physics (enthalpy) ' lies' only if §2 enthalpy-only ≤0. If §2>0 and §3 shows length-")
    print("  driven over-prediction, the enthalpy is RIGHT and a conformational-ENTROPY term is missing.")


if __name__ == "__main__":
    main()
