"""E90 — full pooled (train+test, n=156) within-distribution ranking scorecard.

Proves where we stand among NON-FEP/LIE peptide scorers. Compares, on the combined crystal-65 + the-98
set, our geometry model (+ length router) against every physics single-pose baseline we can compute:
Vina, single-pose MM-GBSA, OpenMM vdW, MJ contact, BSA. Reports pooled LOO r/RMSE/Spearman, per-dataset,
and the held-out (train->test) number. Honest coverage noted where a baseline is dataset-limited.
"""
from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PROD = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
        "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density",
        "cys_frac"]
SHORT = ["bsa_hyd", "mj_contact", "strength_bur"]


def loo(rows, cols, lam=1.0, router=False):
    X = np.array([[r[c] for c in cols] for r in rows], float); y = np.array([r["y"] for r in rows])
    pred = np.full(len(rows), np.nan)
    for i in range(len(rows)):
        tr = [j for j in range(len(rows)) if j != i]
        if router and rows[i]["length"] <= 8:
            trb = [rows[j] for j in tr if rows[j]["length"] <= 8]
            sc = SHORT
            Xt = np.array([[r[c] for c in sc] for r in trb], float); yt = np.array([r["y"] for r in trb])
        else:
            sc = cols
            Xt = X[tr]; yt = y[tr]
        ok = ~np.isnan(Xt).any(1); Xt, yt = Xt[ok], yt[ok]
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
        A = np.column_stack([np.ones(len(Xt)), (Xt - mu) / sd]); R = np.eye(A.shape[1]) * lam; R[0, 0] = 0
        w = np.linalg.solve(A.T @ A + R, A.T @ yt)
        xv = np.array([rows[i][c] for c in sc], float)
        pred[i] = np.r_[1.0, (xv - mu) / sd] @ w
    return pred


def stat(p, y):
    m = ~(np.isnan(p) | np.isnan(y))
    if m.sum() < 5:
        return np.nan, np.nan, np.nan
    return (pearsonr(p[m], y[m])[0], spearmanr(p[m], y[m]).statistic,
            float(np.sqrt(np.mean((p[m] - y[m]) ** 2))))


def fitted_r(x, y):
    """|r| of a single score after a sign-free linear fit (fair to backwards-signed physics like Vina)."""
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 5 or np.std(x[m]) == 0:
        return np.nan, np.nan
    r = pearsonr(x[m], y[m])[0]
    return abs(r), spearmanr(x[m], y[m]).statistic


def main():
    src = open(ROOT / "scripts/e80_charged_gap.py").read().split("def main")[0]
    src = src.replace("Path(__file__).resolve().parents[1]", "Path('%s')" % ROOT)
    ns = {}; exec(src, ns)
    rows = ns["load"]()
    e78 = json.load(open("/tmp/e78_dewet.json"))
    e49 = json.load(open("/tmp/e49b_the98.json"))
    bench = {r["pdb"]: r for r in json.load(open(ROOT / "data/benchmark_crystal.json"))}
    electro = {r["seq"]: r for r in (json.loads(l) for l in open(ROOT / "data/electrostatic_decomp_dataset.jsonl"))}
    for r in rows:
        k = ("cr_" if r["ds"] == "cr65" else "98_") + r["pdb"]
        r["length"] = len(e78.get(k, {}).get("seq", ""))
        r["seq"] = e78.get(k, {}).get("seq", "")
        # physics single-pose: Vina for cr65, single-pose MM-GBSA (dg_single) for the98
        if r["ds"] == "cr65":
            r["phys_single"] = bench.get(r["pdb"], {}).get("vina_docked", np.nan)
            r["vina"] = bench.get(r["pdb"], {}).get("vina_docked", np.nan)
        else:
            r["phys_single"] = e49.get(r["pdb"], {}).get("dg_single", np.nan)
            r["vina"] = np.nan
        r["vdw"] = electro.get(r["seq"], {}).get("vdw", np.nan)

    y = np.array([r["y"] for r in rows])
    cr = np.array([r["ds"] == "cr65" for r in rows]); n98 = ~cr
    print(f"=== E90 POOLED within-distribution scorecard (combined train+test, n={len(rows)}) ===")
    print(f"    crystal-65 n={cr.sum()}  the-98 n={n98.sum()}\n")

    # ---- OUR MODELS (pooled LOO) ----
    print("OUR MODELS (pooled leave-one-out):")
    for nm, cols, rt in [("geometry 16-feat", PROD, False),
                         ("geometry + length router", PROD, True)]:
        p = loo(rows, cols, router=rt)
        r, rho, rmse = stat(p, y)
        rc = stat(p[cr], y[cr]); rn = stat(p[n98], y[n98])
        print(f"  {nm:<28} r={r:.3f} ρ={rho:.3f} RMSE={rmse:.2f} | cr65 r={rc[0]:.2f} the98 r={rn[0]:.2f}")

    # held-out (train->test) for the router model
    def rd(p):
        out = []
        for r in csv.DictReader(open(p)):
            for k in r:
                if k not in ("id", "pdb", "dataset", "affinity_type", "seq"):
                    try: r[k] = float(r[k])
                    except Exception: pass
            out.append(r)
        return out
    tr = rd(ROOT / "data/pooled_benchmark_train.csv"); te = rd(ROOT / "data/pooled_benchmark_test.csv")

    def fit(rows_, cols):
        X = np.array([[r[c] for c in cols] for r in rows_], float); yy = np.array([r["y"] for r in rows_])
        ok = ~np.isnan(X).any(1); X, yy = X[ok], yy[ok]; mu, sd = X.mean(0), X.std(0) + 1e-9
        A = np.column_stack([np.ones(len(X)), (X - mu) / sd]); R = np.eye(A.shape[1]); R[0, 0] = 0
        return mu, sd, np.linalg.solve(A.T @ A + R, A.T @ yy)

    def pr1(r, cols, p): mu, sd, w = p; x = np.array([r[c] for c in cols], float); return float(np.r_[1, (x - mu) / sd] @ w)
    gp = fit(tr, PROD); sp = fit([r for r in tr if r["length"] <= 8], SHORT); rp = fit([r for r in tr if r["length"] > 8], PROD)
    yte = np.array([r["y"] for r in te])
    prt = np.array([pr1(r, SHORT, sp) if r["length"] <= 8 else pr1(r, PROD, rp) for r in te])
    rh = stat(prt, yte)
    print(f"  {'geometry+router (HELD-OUT)':<28} r={rh[0]:.3f} ρ={rh[1]:.3f} RMSE={rh[2]:.2f}  (true train->test, n={len(te)})")

    # ---- PHYSICS / BASELINE SCORERS (fitted |r|, fair to backwards signs) ----
    print("\nPHYSICS & FEATURE BASELINES (fitted |r|; coverage noted):")
    def col(f, mask=None):
        x = np.array([r.get(f, np.nan) for r in rows], float)
        return x if mask is None else x[mask]
    rows_arr = rows
    baselines = [
        ("Vina (cr65 only)", "vina", cr),
        ("MM-GBSA single-pose (the98)", "phys_single", n98),
        ("physics single-pose (pooled)", "phys_single", None),
        ("OpenMM vdW packing", "vdw", None),
        ("MJ contact energy", "mj_contact", None),
        ("BSA hydrophobic", "bsa_hyd", None),
        ("strength_bur (SKEMPI)", "strength_bur", None),
    ]
    for nm, f, mask in baselines:
        x = col(f, mask); yy = y if mask is None else y[mask]
        r, rho = fitted_r(x, yy)
        nval = (~np.isnan(x)).sum()
        print(f"  {nm:<30} |r|={r:.3f} ρ={rho:+.3f}  (n={nval})")

    print("\nLITERATURE PEERS (non-FEP/LIE, from docs): PPI-Affinity r=0.55 | FlexPepDock 0.59 (within-target)")
    print("  | Rosetta ref2015 unrelaxed 0.16 | Vina-alone fitted 0.53. FEP/LIE (the ceiling): 0.8-0.9.")


if __name__ == "__main__":
    main()
