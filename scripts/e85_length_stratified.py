"""E85 — full length-stratified autopsy (Ram's ask): r/RMSE/slope + error bars per length bin, and what
feature/type drives the per-bin error. ANALYSIS ONLY — no wiring.

Bins (from distribution): short<=8 (n22), med 9-12 (n78), long 13-18 (n34), vlong>=19 (n22).
For each bin, pooled-LOO production model:
  r (+ bootstrap 95% CI), Spearman, RMSE, slope (true~pred; <1 = compression), mean signed residual
  (over/under prediction), and the residual breakdown by affinity type + charge.
Then: per-bin residual-feature correlations (what we are MISSING in each bin) and a test of whether
length-AWARE modeling (per-bin calibration + length-interaction terms) would help out-of-sample.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]


def main():
    src = open(ROOT / "scripts/e80_charged_gap.py").read().split("def main")[0]
    src = src.replace("Path(__file__).resolve().parents[1]", "Path('%s')" % ROOT)
    ns = {}; exec(src, ns)
    rows = ns["load"](); PROD = ns["PROD"]
    e78 = json.load(open("/tmp/e78_dewet.json"))
    bench = {r["pdb"]: r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    for r in rows:
        k = ("cr_" if r["ds"] == "cr65" else "98_") + r["pdb"]
        r["length"] = len(e78.get(k, {}).get("seq", ""))
        r["atype"] = bench.get(r["pdb"], {}).get("affinity_type", "Kd") if r["ds"] == "cr65" else "Kd"

    pred = ns["loo_pred"](rows, PROD)
    for r, p in zip(rows, pred):
        r["_pred"] = p; r["_res"] = r["y"] - p
    y = np.array([r["y"] for r in rows])

    BINS = [("short", 0, 8), ("med", 9, 12), ("long", 13, 18), ("vlong", 19, 99)]

    def boot_r(p, yy, B=2000):
        rs = []
        n = len(p)
        for _ in range(B):
            i = np.random.randint(0, n, n)
            if np.std(p[i]) > 0 and np.std(yy[i]) > 0:
                rs.append(pearsonr(p[i], yy[i])[0])
        return np.percentile(rs, [2.5, 97.5]) if rs else (np.nan, np.nan)

    print("=== E85 length-stratified performance (pooled-LOO production model) ===")
    print(f"{'bin':<7}{'n':>4}{'len':>7}{'r':>7}{'  95% CI':>16}{'rho':>7}{'RMSE':>7}{'slope':>7}"
          f"{'meanRes':>9}")
    np.random.seed(0)
    for nm, lo, hi in BINS:
        m = np.array([lo <= r["length"] <= hi for r in rows])
        p, yy = pred[m], y[m]
        r = pearsonr(p, yy)[0]; rho = spearmanr(p, yy).statistic
        rmse = np.sqrt(np.mean((p - yy) ** 2)); slope = np.polyfit(p, yy, 1)[0]
        ci = boot_r(p, yy); mres = np.mean(yy - p)
        lmean = np.mean([rr["length"] for rr in rows if lo <= rr["length"] <= hi])
        print(f"  {nm:<5}{m.sum():>4}{lmean:>7.1f}{r:>7.3f}  [{ci[0]:+.2f},{ci[1]:+.2f}]"
              f"{rho:>7.3f}{rmse:>7.2f}{slope:>7.2f}{mres:>+9.2f}")
    print("  (slope<1 = compression; meanRes>0 = true stronger than pred = we UNDER-predict the bin)")

    # over/under split within each bin (strong vs weak half)
    print("\n=== over/under prediction within each bin (signed residual, strong vs weak half) ===")
    for nm, lo, hi in BINS:
        sub = [r for r in rows if lo <= r["length"] <= hi]
        yy = np.array([r["y"] for r in sub]); med = np.median(yy)
        st = [r for r in sub if r["y"] <= med]; wk = [r for r in sub if r["y"] > med]
        print(f"  {nm:<6} strong-half res={np.mean([r['_res'] for r in st]):+.2f}  "
              f"weak-half res={np.mean([r['_res'] for r in wk]):+.2f}  "
              f"(Ki frac={np.mean([r['atype']!='Kd' for r in sub]):.2f}, "
              f"chg frac={np.mean([abs(r['net_charge'])>=2 for r in sub]):.2f})")

    # what feature drives the error in each bin
    feats = PROD + ["net_charge", "net_dewet", "polar_desolv", "rg_per_L", "length"]
    print("\n=== per-bin residual drivers: top |corr(feature, residual)| (what we MISS in that bin) ===")
    for nm, lo, hi in BINS:
        sub = [r for r in rows if lo <= r["length"] <= hi]
        e = np.array([r["_res"] for r in sub])
        cors = []
        for f in feats:
            x = np.array([r.get(f, np.nan) for r in sub], float); mk = ~np.isnan(x)
            if mk.sum() > 6 and np.std(x[mk]) > 0:
                cors.append((f, pearsonr(x[mk], e[mk])[0]))
        cors.sort(key=lambda t: -abs(t[1]))
        top = "  ".join(f"{f}={c:+.2f}" for f, c in cors[:4])
        print(f"  {nm:<6} {top}")

    # does length-AWARE modeling help out-of-sample? (per-bin calib vs global, 5-fold within pooled)
    print("\n=== would LENGTH-AWARE modeling help? (5-fold CV, pooled) ===")

    def cv(cols, perbin=False, K=5):
        idx = np.arange(len(rows)); np.random.seed(1); np.random.shuffle(idx)
        folds = [idx[i::K] for i in range(K)]
        preds = np.zeros(len(rows))
        for k in range(K):
            te = folds[k]; tr = np.array([i for i in idx if i not in set(te)])

            def fit(tr_rows):
                X = np.array([[r[c] for c in cols] for r in tr_rows], float)
                yt = np.array([r["y"] for r in tr_rows])
                ok = ~np.isnan(X).any(1); X, yt = X[ok], yt[ok]
                mu, sd = X.mean(0), X.std(0) + 1e-9
                A = np.column_stack([np.ones(len(X)), (X - mu) / sd]); R = np.eye(A.shape[1]); R[0, 0] = 0
                return mu, sd, np.linalg.solve(A.T @ A + R, A.T @ yt)
            if perbin:
                for nm, lo, hi in BINS:
                    trb = [rows[i] for i in tr if lo <= rows[i]["length"] <= hi]
                    teb = [i for i in te if lo <= rows[i]["length"] <= hi]
                    if len(trb) < 8 or not teb:
                        for i in teb:
                            preds[i] = np.mean([r["y"] for r in [rows[j] for j in tr]])
                        continue
                    mu, sd, w = fit(trb)
                    for i in teb:
                        xv = np.array([rows[i][c] for c in cols], float)
                        preds[i] = np.r_[1.0, (xv - mu) / sd] @ w
            else:
                mu, sd, w = fit([rows[i] for i in tr])
                for i in te:
                    xv = np.array([rows[i][c] for c in cols], float)
                    preds[i] = np.r_[1.0, (xv - mu) / sd] @ w
        return pearsonr(preds, y)[0], np.sqrt(np.mean((preds - y) ** 2))

    rL, eL = cv(PROD)
    rB, eB = cv(PROD, perbin=True)
    # length-normalized features: divide extensive features by length
    EXT = ["bsa_hyd", "sasa_hb", "sasa_sb", "mj_contact", "hb_count", "poc_n", "mean_burial"]
    for r in rows:
        for f in EXT:
            r[f + "_pL"] = r[f] / max(1, r["length"])
    rN, eN = cv([c + "_pL" if c in EXT else c for c in PROD] + ["length"])
    print(f"  global PROD:          r={rL:+.3f} RMSE={eL:.2f}")
    print(f"  per-bin PROD (sep):   r={rB:+.3f} RMSE={eB:.2f}")
    print(f"  length-normalized:    r={rN:+.3f} RMSE={eN:.2f}")
    print("  (if per-bin or length-normalized beats global, length-aware scoring is justified.)")


if __name__ == "__main__":
    main()
