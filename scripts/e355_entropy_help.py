"""E355 — does the confinement entropy term help ABSOLUTE kcal/mol? (the direct A-vs-B test)

Reads whatever E354 has computed (data/e354_confinement.json) and asks the practical question BEFORE the
residual-shape gate: adding TΔS_config to the scorer, does absolute-ΔG r/MAE/RMSE improve?
  A (scorer features)  vs  B (scorer features + TΔS_config)     leave-one-out grouped
Plus the residual correlation (the mechanistic signal). Small-n honest: reports both, flags if n<15.

Run: OMP_NUM_THREADS=1 python scripts/e355_entropy_help.py
"""
from __future__ import annotations
import json
import numpy as np
from scipy.stats import pearsonr

FULL = ["poc_n","poc_f_hyd","poc_f_arom","poc_net","poc_eis","bsa_hyd","sasa_hb","sasa_sb","arom_cc",
        "hb_count","mj_contact","strength_bur","rg_per_L","org_density","cys_frac","mean_burial"]


def main():
    try:
        ent = json.load(open("data/e354_confinement.json"))
    except FileNotFoundError:
        print("no data/e354_confinement.json yet — gate still on its first complex"); return
    ent = [e for e in ent if e.get("tds") is not None and np.isfinite(e["tds"])]
    if len(ent) < 5:
        print(f"only {len(ent)} complexes with TΔS — need >=5"); return
    pep = {json.loads(x)["pdb"]: json.loads(x) for x in open("data/pdbbind_peptides.jsonl")}
    allrows = [json.loads(x) for x in open("data/pdbbind_peptides.jsonl")]

    # scorer residual from a FULL-set leakage-free scorer (so residual is well-defined), read on our subset
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    Xall = np.array([[float(r[f]) for f in FULL] for r in allrows]); yall = np.array([float(r["y"]) for r in allrows])
    gall = np.array([hash(r["seq"][:4]) % 100000 for r in allrows])
    oof = np.full(len(yall), np.nan)
    for tr, te in GroupKFold(8).split(Xall, yall, gall):
        m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=15, random_state=0)
        m.fit(Xall[tr], yall[tr]); oof[te] = m.predict(Xall[te])
    resid = {allrows[i]["pdb"]: float(yall[i] - oof[i]) for i in range(len(allrows))}

    rows = [(e["pdb"], pep[e["pdb"]], e["tds"], float(e["y"])) for e in ent if e["pdb"] in pep]
    y = np.array([r[3] for r in rows]); tds = np.array([r[2] for r in rows])
    rs = np.array([resid[r[0]] for r in rows])
    Xs = np.array([[float(r[1][f]) for f in FULL] for r in rows])
    groups = np.array([hash(r[1]["seq"][:4]) % 100000 for r in rows])
    n = len(rows)
    print(f"=== E355 entropy-help test  (n={n}) ===")
    print(f"TΔS_config: mean={tds.mean():+.2f} std={tds.std():.2f} range [{tds.min():+.1f},{tds.max():+.1f}]")

    # 1) the direct absolute-help test: A (scorer) vs B (+TΔS), leave-one-out
    def loo(X):
        oof = np.full(n, np.nan)
        k = min(6, n)
        for tr, te in GroupKFold(k).split(X, y, groups):
            m = HistGradientBoostingRegressor(max_depth=2, learning_rate=0.05, max_iter=200, min_samples_leaf=5, random_state=0)
            m.fit(X[tr], y[tr]); oof[te] = m.predict(X[te])
        return oof
    A = loo(Xs); B = loo(np.hstack([Xs, tds.reshape(-1, 1)]))
    for name, p in (("A scorer      ", A), ("B scorer+TΔS  ", B)):
        r, mae, rmse = pearsonr(p, y)[0], np.mean(np.abs(p - y)), np.sqrt(np.mean((p - y) ** 2))
        print(f"  {name} r={r:+.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}")

    # 2) direct correlations (more powerful at small n than the GBT A/B)
    print(f"  corr(TΔS, y)                 = {pearsonr(tds,y)[0]:+.3f}")
    print(f"  corr(TΔS, scorer_residual)   = {pearsonr(tds,rs)[0]:+.3f}   <- if >|0.25|, entropy has residual shape")
    # simple linear: does residual − a*TΔS shrink? best linear TΔS coefficient
    if tds.std() > 1e-6:
        a = np.cov(tds, rs)[0, 1] / np.var(tds)
        rmse_before = np.sqrt(np.mean(rs**2)); rmse_after = np.sqrt(np.mean((rs - a*tds)**2))
        print(f"  residual RMSE: {rmse_before:.3f} -> {rmse_after:.3f} after subtracting best-linear TΔS (Δ={rmse_before-rmse_after:+.3f})")
    if n < 15:
        print(f"  [n={n} is small — treat as preliminary; full gate is n=30]")


if __name__ == "__main__":
    main()
