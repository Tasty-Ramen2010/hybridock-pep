"""E59 — full residual autopsy of the ABSOLUTE scoring function on the-98.

Question (Ram): where does our absolute ΔG fail on the-98, broken down by peptide type / #residues /
burial / strength / length / SS — and do we OVER- or UNDER-predict? Then the decisive test: per
complex, find the single constant c_i that makes pred_i + c_i land within ~1 kcal/mol of experiment
(c_i = y_i − pred_i, the per-complex offset). Is c_i PREDICTABLE from peptide features (linear OR
nonlinear)? If yes, that feature IS the missing systematic term in the scoring function.

Predictor = MM-GBSA single-point (dg_single), globally calibrated to kcal/mol scale via OLS pred=a·g+b
(a real scoring function, not per-complex cheating). Also reports ensemble ⟨E_int⟩ for contrast.
Data: /tmp/e49b_the98.json (y, dg_single, e_int_mean, e_int_std, minus_tds, seq, L, cf)
      /tmp/e28_feats.json  (mj_contact, bsa_hyd, poc_n, poc_net, arom_cc, sasa_hb/sb, ss)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
      "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
      "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}


def seqfeat(seq):
    L = max(1, len(seq))
    return dict(
        net_charge=seq.count("K") + seq.count("R") - seq.count("D") - seq.count("E"),
        abs_net_charge=abs(seq.count("K") + seq.count("R") - seq.count("D") - seq.count("E")),
        charged_frac=sum(c in "DEKR" for c in seq) / L,
        hyd_frac=sum(c in "AILMFVWC" for c in seq) / L,
        arom_frac=sum(c in "FWYH" for c in seq) / L,
        bulky_frac=sum(c in "FWYLIM" for c in seq) / L,
        pro_frac=seq.count("P") / L, gly_frac=seq.count("G") / L,
        progly_frac=(seq.count("P") + seq.count("G")) / L,
        term_flex=sum(c in "GSDEKR" for c in (seq[:2] + seq[-2:])) / 4.0,
        mean_kd=float(np.mean([KD.get(c, 0) for c in seq])),
    )


def fit_global(g, y):
    """OLS pred = a*g + b -> kcal/mol scale."""
    a, b = np.polyfit(g, y, 1)
    return a * np.asarray(g) + b, (a, b)


def grp(rows, key, label, fn):
    lo = [r for r in rows if not fn(r)]
    hi = [r for r in rows if fn(r)]
    if len(lo) < 4 or len(hi) < 4:
        return
    for nm, sub in [(f"{label}=LO", lo), (f"{label}=HI", hi)]:
        err = np.array([r["pred"] - r["y"] for r in sub])  # +ve => pred less negative => UNDER-bind
        print(f"  {nm:<22} n={len(sub):>3}  meanErr(pred−exp)={err.mean():+5.2f}  "
              f"|err|={np.abs(err).mean():4.2f}  "
              f"{'UNDER-predicts affinity' if err.mean() > 0.3 else 'OVER-predicts affinity' if err.mean() < -0.3 else 'balanced'}")


def main():
    e49 = json.loads(Path("/tmp/e49b_the98.json").read_text())
    e28 = json.loads(Path("/tmp/e28_feats.json").read_text())
    rows = []
    for k, v in e49.items():
        f28 = e28.get(k, {})
        r = dict(key=k, y=v["y"], g=v["dg_single"], eint=v["e_int_mean"],
                 eint_std=v.get("e_int_std", np.nan), mtds=v.get("minus_tds", np.nan),
                 L=v["L"], seq=v["seq"],
                 mj=f28.get("mj_contact", np.nan), bsa_hyd=f28.get("bsa_hyd", np.nan),
                 poc_n=f28.get("poc_n", np.nan), poc_net=f28.get("poc_net", np.nan),
                 arom_cc=f28.get("arom_cc", np.nan), sasa_hb=f28.get("sasa_hb", np.nan),
                 sasa_sb=f28.get("sasa_sb", np.nan), ss=f28.get("ss", "?"))
        r.update(seqfeat(v["seq"]))
        r["bsa_per_L"] = r["bsa_hyd"] / max(1, r["L"]) if not np.isnan(r["bsa_hyd"]) else np.nan
        r["mj_per_L"] = r["mj"] / max(1, r["L"]) if not np.isnan(r["mj"]) else np.nan
        rows.append(r)
    y = np.array([r["y"] for r in rows])
    g = np.array([r["g"] for r in rows])
    pred, (a, b) = fit_global(g, y)
    for i, r in enumerate(rows):
        r["pred"] = float(pred[i])
        r["resid"] = float(r["y"] - r["pred"])   # the per-complex offset c_i to add
        r["abserr"] = abs(r["resid"])

    print(f"=== E59 absolute-scoring residual autopsy on the-98 (n={len(rows)}) ===")
    print(f"Predictor: MM-GBSA single-point, global OLS calib pred={a:.4f}·g+{b:.2f}")
    print(f"  baseline Pearson(pred,exp)={pearsonr(pred,y)[0]:+.3f}  Spearman={spearmanr(pred,y).statistic:+.3f}")
    print(f"  RMSE={np.sqrt(np.mean((pred-y)**2)):.2f} kcal/mol   MAE={np.mean(np.abs(pred-y)):.2f}")
    within1 = np.mean(np.abs(pred - y) < 1.0) * 100
    within2 = np.mean(np.abs(pred - y) < 2.0) * 100
    print(f"  within 1 kcal/mol: {within1:.0f}%   within 2: {within2:.0f}%")
    print(f"  exp ΔG range [{y.min():.1f},{y.max():.1f}]  pred range [{pred.min():.1f},{pred.max():.1f}]")

    print("\n=== OVER/UNDER-PREDICTION by factor (err = pred − exp; +ve = under-predicts affinity) ===")
    med = lambda key: np.nanmedian([r[key] for r in rows])
    grp(rows, "L", "length", lambda r: r["L"] > med("L"))
    grp(rows, "y", "strength", lambda r: r["y"] < med("y"))         # HI = stronger (more −ve)
    grp(rows, "charged_frac", "charged_frac", lambda r: r["charged_frac"] > med("charged_frac"))
    grp(rows, "abs_net_charge", "|net charge|", lambda r: r["abs_net_charge"] > 1.5)
    grp(rows, "hyd_frac", "hydrophobic", lambda r: r["hyd_frac"] > med("hyd_frac"))
    grp(rows, "bsa_per_L", "burial/res", lambda r: (r["bsa_per_L"] > med("bsa_per_L")) if not np.isnan(r["bsa_per_L"]) else False)
    grp(rows, "arom_frac", "aromatic", lambda r: r["arom_frac"] > med("arom_frac"))
    grp(rows, "progly_frac", "pro/gly", lambda r: r["progly_frac"] > 0.1)
    for s in ["HELIX", "SHEET", "LOOP"]:
        sub = [r for r in rows if str(r["ss"]).upper().startswith(s[:4])]
        if len(sub) >= 4:
            err = np.array([r["pred"] - r["y"] for r in sub])
            print(f"  ss={s:<18} n={len(sub):>3}  meanErr={err.mean():+5.2f}  |err|={np.abs(err).mean():4.2f}")

    print("\n=== RESIDUAL ~ FEATURE correlation (does the offset c_i track a feature? LINEAR) ===")
    resid = np.array([r["resid"] for r in rows])
    feats = ["L", "net_charge", "abs_net_charge", "charged_frac", "hyd_frac", "arom_frac",
             "bulky_frac", "progly_frac", "term_flex", "mean_kd", "mj", "mj_per_L",
             "bsa_hyd", "bsa_per_L", "poc_n", "poc_net", "eint_std", "mtds"]
    res_corr = []
    for f in feats:
        x = np.array([r[f] for r in rows], dtype=float)
        m = ~np.isnan(x)
        if m.sum() < 20 or np.std(x[m]) < 1e-9:
            continue
        pr = pearsonr(x[m], resid[m])[0]
        sp = spearmanr(x[m], resid[m]).statistic
        res_corr.append((f, pr, sp))
    for f, pr, sp in sorted(res_corr, key=lambda t: -abs(t[2])):
        flag = "  <== signal" if abs(sp) > 0.3 else ""
        print(f"  {f:<14} Pearson={pr:+.3f}  Spearman={sp:+.3f}{flag}")

    print("\n=== NONLINEAR: can a model predict the offset c_i? (leave-one-out) ===")
    F = [f for f, _, _ in res_corr]
    X = np.array([[r[f] for f in F] for r in rows], dtype=float)
    col_ok = ~np.isnan(X).any(0)
    X = X[:, col_ok]
    Fok = [f for f, c in zip(F, col_ok) if c]
    yv = resid
    # linear ridge LOO R2
    def loo_r2(model_fn):
        preds = np.zeros(len(yv))
        for i in range(len(yv)):
            tr = np.arange(len(yv)) != i
            preds[i] = model_fn(X[tr], yv[tr], X[i:i + 1])[0]
        ss_res = np.sum((yv - preds) ** 2)
        ss_tot = np.sum((yv - yv.mean()) ** 2)
        return 1 - ss_res / ss_tot, spearmanr(preds, yv).statistic
    def ridge(Xt, yt, Xe):
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
        A = np.column_stack([np.ones(len(Xt)), (Xt - mu) / sd])
        R = 3.0 * np.eye(A.shape[1]); R[0, 0] = 0
        w = np.linalg.solve(A.T @ A + R, A.T @ yt)
        return np.column_stack([np.ones(len(Xe)), (Xe - mu) / sd]) @ w
    r2_lin, sp_lin = loo_r2(ridge)
    print(f"  ridge (linear, all feats) LOO R²={r2_lin:+.3f}  Spearman(pred c, true c)={sp_lin:+.3f}")
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        def gbt(Xt, yt, Xe):
            m = GradientBoostingRegressor(n_estimators=120, max_depth=2, learning_rate=0.05)
            m.fit(Xt, yt); return m.predict(Xe)
        r2_gbt, sp_gbt = loo_r2(gbt)
        print(f"  GBT  (nonlinear)          LOO R²={r2_gbt:+.3f}  Spearman={sp_gbt:+.3f}")
        # feature importance on full fit
        m = GradientBoostingRegressor(n_estimators=120, max_depth=2, learning_rate=0.05).fit(X, yv)
        imp = sorted(zip(Fok, m.feature_importances_), key=lambda t: -t[1])[:6]
        print("  GBT top features for the offset:", ", ".join(f"{f}={i:.2f}" for f, i in imp))
    except Exception as e:  # noqa: BLE001
        print("  GBT skipped:", str(e)[:60])
    print("\n  >> If LOO R²>0 and a feature has |Spearman|>0.3, the offset is SYSTEMATIC = missing term.")
    print("  >> If LOO R²<=0, the per-complex error is NOISE (pose/measurement), not a learnable term.")


if __name__ == "__main__":
    main()
