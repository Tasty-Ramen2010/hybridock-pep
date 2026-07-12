"""E60 — Ram's challenge done right: (1) are the over/under-predicted a DISTINCT CLASS (not a gradient)?
(2) does adding a feature-correction IMPROVE OUT-OF-SAMPLE ranking (not just fit the residual)?
(3) can a CHEAP model-space ensemble (blend of scoring fns) un-compress the dynamic range?

Key fix over e59: the honest test of "a correction that scales" is whether y ~ pred + F beats y ~ pred
in LEAVE-ONE-OUT correlation with the REAL ΔG. Residual-R² was circular (residual≈strength by
construction). Also test class-conditional constants and multi-score blends.

Data: /tmp/e49b_the98.json + /tmp/e28_feats.json (cached). No new compute.
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
        progly_frac=(seq.count("P") + seq.count("G")) / L,
        mean_kd=float(np.mean([KD.get(c, 0) for c in seq])),
        L=len(seq),
    )


def load():
    e49 = json.loads(Path("/tmp/e49b_the98.json").read_text())
    e28 = json.loads(Path("/tmp/e28_feats.json").read_text())
    rows = []
    for k, v in e49.items():
        f = e28.get(k, {})
        r = dict(key=k, y=v["y"], mmgbsa=v["dg_single"], eint=v["e_int_mean"],
                 eint_std=v.get("e_int_std", np.nan), mtds=v.get("minus_tds", np.nan), seq=v["seq"],
                 mj=f.get("mj_contact", np.nan), bsa_hyd=f.get("bsa_hyd", np.nan),
                 poc_n=f.get("poc_n", np.nan), poc_net=f.get("poc_net", np.nan),
                 arom_cc=f.get("arom_cc", np.nan), sasa_hb=f.get("sasa_hb", np.nan),
                 sasa_sb=f.get("sasa_sb", np.nan), ss=str(f.get("ss", "?")))
        r.update(seqfeat(v["seq"]))
        rows.append(r)
    return rows


def loo_pearson(rows, cols):
    """y ~ OLS(cols) leave-one-out, return Pearson & Spearman vs real y."""
    X = np.array([[r[c] for c in cols] for r in rows], dtype=float)
    y = np.array([r["y"] for r in rows])
    ok = ~np.isnan(X).any(1)
    X, y2 = X[ok], y[ok]
    preds = np.zeros(len(y2))
    for i in range(len(y2)):
        tr = np.arange(len(y2)) != i
        Xt, yt = X[tr], y2[tr]
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
        A = np.column_stack([np.ones(len(Xt)), (Xt - mu) / sd])
        Rr = 1.0 * np.eye(A.shape[1]); Rr[0, 0] = 0
        w = np.linalg.solve(A.T @ A + Rr, A.T @ yt)
        preds[i] = np.r_[1, (X[i] - mu) / sd] @ w
    return pearsonr(preds, y2)[0], spearmanr(preds, y2).statistic, len(y2)


def main():
    rows = load()
    y = np.array([r["y"] for r in rows])
    g = np.array([r["mmgbsa"] for r in rows])
    a, b = np.polyfit(g, y, 1); pred = a * g + b
    resid = y - pred
    print(f"=== E60 (n={len(rows)}) baseline MMGBSA LOO ===")
    p0, s0, _ = loo_pearson(rows, ["mmgbsa"])
    print(f"  y ~ mmgbsa: LOO Pearson={p0:+.3f} Spearman={s0:+.3f}")
    print(f"  corr(residual, y) = {pearsonr(resid, y)[0]:+.3f}  <- if ~1, residual IS strength (circular)")

    print("\n=== (1) TAILS: are over/under-predicted a DISTINCT CLASS? ===")
    order = np.argsort(resid)
    print("  --- 8 MOST OVER-predicted (we say too strong; resid>0 means exp weaker) ---")
    print("  --- (resid = exp − pred; very NEGATIVE resid = we overshoot binding) ---")
    for i in order[:8]:
        r = rows[i]
        print(f"   {r['key']:<12} exp={r['y']:+.1f} pred={pred[i]:+.1f} resid={resid[i]:+.1f} "
              f"L={r['L']} netQ={r['net_charge']:+d} hyd={r['hyd_frac']:.2f} arom={r['arom_frac']:.2f} ss={r['ss'][:5]} {r['seq']}")
    print("  --- 8 MOST UNDER-predicted (we say too weak; resid very POSITIVE) ---")
    for i in order[::-1][:8]:
        r = rows[i]
        print(f"   {r['key']:<12} exp={r['y']:+.1f} pred={pred[i]:+.1f} resid={resid[i]:+.1f} "
              f"L={r['L']} netQ={r['net_charge']:+d} hyd={r['hyd_frac']:.2f} arom={r['arom_frac']:.2f} ss={r['ss'][:5]} {r['seq']}")

    print("\n=== (2) DOES ADDING A FEATURE IMPROVE OUT-OF-SAMPLE? (the honest 'does it scale' test) ===")
    print(f"  baseline y~mmgbsa LOO Pearson={p0:+.3f}")
    feats = ["L", "net_charge", "abs_net_charge", "charged_frac", "hyd_frac", "arom_frac",
             "bulky_frac", "progly_frac", "mean_kd", "mj", "bsa_hyd", "poc_n", "poc_net",
             "eint", "eint_std", "mtds", "sasa_hb", "sasa_sb", "arom_cc"]
    gains = []
    for f in feats:
        if all(np.isnan(r[f]) if isinstance(r[f], float) else False for r in rows):
            continue
        try:
            p, s, n = loo_pearson(rows, ["mmgbsa", f])
            gains.append((f, p, p - p0))
        except Exception:
            continue
    for f, p, dg in sorted(gains, key=lambda t: -t[2]):
        flag = "  <== IMPROVES" if dg > 0.03 else ("  (hurts)" if dg < -0.03 else "")
        print(f"  y ~ mmgbsa + {f:<13} LOO Pearson={p:+.3f}  Δ={dg:+.3f}{flag}")

    print("\n=== (3) CHEAP MODEL-SPACE ENSEMBLE (blend scoring fns — un-compress without MD?) ===")
    for nm, cols in [("mmgbsa", ["mmgbsa"]),
                     ("mmgbsa+eint", ["mmgbsa", "eint"]),
                     ("mmgbsa+mj", ["mmgbsa", "mj"]),
                     ("mmgbsa+eint+mj", ["mmgbsa", "eint", "mj"]),
                     ("mmgbsa+mj+bsa+poc_n", ["mmgbsa", "mj", "bsa_hyd", "poc_n"]),
                     ("ALL physics scores", ["mmgbsa", "eint", "mj", "bsa_hyd", "poc_n", "sasa_hb", "sasa_sb"])]:
        try:
            p, s, n = loo_pearson(rows, cols)
            print(f"  {nm:<24} LOO Pearson={p:+.3f} Spearman={s:+.3f} (n={n})")
        except Exception as e:  # noqa: BLE001
            print(f"  {nm}: {str(e)[:40]}")

    print("\n=== (4) CLASS-CONDITIONAL recalibration (per-SS / per-charge own slope+intercept, LOO) ===")
    # split by a binary class, fit separate OLS per class, LOO within the pooled scheme
    for clsname, clsfn in [("by SS(helix/sheet)", lambda r: r["ss"][:5]),
                           ("by charged(net!=0)", lambda r: "chg" if r["net_charge"] != 0 else "neu"),
                           ("by length(>=12)", lambda r: "long" if r["L"] >= 12 else "short")]:
        classes = sorted({clsfn(r) for r in rows})
        preds = np.full(len(rows), np.nan)
        for i in range(len(rows)):
            c = clsfn(rows[i])
            tr = [j for j in range(len(rows)) if j != i and clsfn(rows[j]) == c]
            if len(tr) < 6:
                tr = [j for j in range(len(rows)) if j != i]  # fall back to global
            gg = np.array([rows[j]["mmgbsa"] for j in tr]); yy = np.array([rows[j]["y"] for j in tr])
            aa, bb = np.polyfit(gg, yy, 1)
            preds[i] = aa * rows[i]["mmgbsa"] + bb
        print(f"  {clsname:<22} LOO Pearson={pearsonr(preds, y)[0]:+.3f} (classes={classes})")

    print("\n  >> A feature that lifts (2) out-of-sample, or a class recalib that beats baseline, = REAL.")


if __name__ == "__main__":
    main()
