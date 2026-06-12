"""E86 — (A) length x feature interaction model + (B) deep dive into WHY short peptides are r=0.

(A) Single pooled model with feature x length-bin interaction terms (NOT separate scorers, which starved
    the data). Each feature gets a global coefficient + a per-bin delta, so the model can up-weight
    hydrophobic anchors for short and the extendedness penalty for vlong, all trained on full n.
    Test out-of-sample on the unbiased pooled split (train.csv/test.csv) vs global PROD.

(B) Short-peptide (<=8, n22) autopsy: WHY r=0?
    1. dynamic range: std of each feature on shorts vs the rest (collapsed range -> no signal possible)
    2. per-feature corr with ΔG on shorts ALONE (is ANY feature predictive, just masked in the global fit?)
    3. ΔG spread + label composition (maybe shorts just have tiny ΔG variance = ceiling on r)
    4. can a short-ONLY fit (or a couple short-specific features) recover signal?
ANALYSIS ONLY — Ram said don't implement confidence numbers yet; this shows what's possible.
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
BINS = [("short", 0, 8), ("med", 9, 12), ("long", 13, 18), ("vlong", 19, 99)]


def load_rows():
    src = open(ROOT / "scripts/e80_charged_gap.py").read().split("def main")[0]
    src = src.replace("Path(__file__).resolve().parents[1]", "Path('%s')" % ROOT)
    ns = {}; exec(src, ns)
    rows = ns["load"]()
    e78 = json.load(open("/tmp/e78_dewet.json"))
    for r in rows:
        k = ("cr_" if r["ds"] == "cr65" else "98_") + r["pdb"]
        r["length"] = len(e78.get(k, {}).get("seq", ""))
        r["net_dewet"] = e78.get(k, {}).get("net_dewet", np.nan)
        r["polar_desolv"] = e78.get(k, {}).get("polar_desolv", np.nan)
        r["seq"] = e78.get(k, {}).get("seq", "")
    return rows, ns


def binid(L):
    for i, (nm, lo, hi) in enumerate(BINS):
        if lo <= L <= hi:
            return i, nm
    return 1, "med"


# ---------- (A) length-interaction model ----------
def partA(rows, ns):
    print("=== (A) length x feature INTERACTION model (single pooled, full-n) ===")
    # build design: PROD + (PROD * 1[bin]) for short & vlong (the off-calibrated bins)
    def design(rs, inter_bins):
        base = np.array([[r[c] for c in PROD] for r in rs], float)
        cols = list(base.T)
        for b in inter_bins:
            mask = np.array([1.0 if binid(r["length"])[1] == b else 0.0 for r in rs])
            for j, c in enumerate(PROD):
                cols.append(base[:, j] * mask)
        return np.column_stack(cols)

    y = np.array([r["y"] for r in rows])

    def cv(design_fn, K=5, seed=1):
        idx = np.arange(len(rows)); np.random.seed(seed); np.random.shuffle(idx)
        folds = [idx[i::K] for i in range(K)]
        pred = np.zeros(len(rows))
        for k in range(K):
            te = set(folds[k]); tr = [i for i in idx if i not in te]
            Xtr = design_fn([rows[i] for i in tr]); ytr = y[tr]
            ok = ~np.isnan(Xtr).any(1); Xtr, ytr = Xtr[ok], ytr[ok]
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            A = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd]); R = np.eye(A.shape[1]) * 2; R[0, 0] = 0
            w = np.linalg.solve(A.T @ A + R, A.T @ ytr)
            Xte = design_fn([rows[i] for i in folds[k]])
            P = np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd]) @ w
            for jj, i in enumerate(folds[k]):
                pred[i] = P[jj]
        return pred

    for nm, fn in [("global PROD", lambda rs: np.array([[r[c] for c in PROD] for r in rs], float)),
                   ("+ short interaction", lambda rs: design(rs, ["short"])),
                   ("+ short+vlong interaction", lambda rs: design(rs, ["short", "vlong"])),
                   ("+ all-bin interaction", lambda rs: design(rs, ["short", "long", "vlong"]))]:
        p = cv(fn)
        r = pearsonr(p, y)[0]; rmse = np.sqrt(np.mean((p - y) ** 2))
        # per-bin r
        perbin = []
        for bn, lo, hi in BINS:
            m = np.array([lo <= rr["length"] <= hi for rr in rows])
            perbin.append(f"{bn}={pearsonr(p[m], y[m])[0]:+.2f}")
        print(f"  {nm:<28} r={r:+.3f} RMSE={rmse:.2f}  [{'  '.join(perbin)}]")
    print("  (does any interaction lift the SHORT r without hurting the rest? if not, shorts are noise.)")


# ---------- (B) short-peptide deep dive ----------
def partB(rows, ns):
    print("\n=== (B) WHY are short peptides (<=8) r=0? deep dive (n=22) ===")
    sh = [r for r in rows if r["length"] <= 8]
    rest = [r for r in rows if r["length"] > 8]
    y = np.array([r["y"] for r in sh])
    print(f"  short n={len(sh)}  ΔG: mean={y.mean():.2f} std={y.std():.2f}  range=[{y.min():.1f},{y.max():.1f}]")
    yr = np.array([r["y"] for r in rest])
    print(f"  rest  n={len(rest)} ΔG: mean={yr.mean():.2f} std={yr.std():.2f}")
    print("  (if short ΔG std << rest, low r is partly a NARROW-TARGET ceiling, not pure feature failure.)")

    print("\n  1. FEATURE DYNAMIC RANGE on shorts (std_short/std_rest; <1 = collapsed = can't carry signal):")
    feats = PROD + ["net_dewet", "polar_desolv", "net_charge"]
    ranges = []
    for f in feats:
        xs = np.array([r.get(f, np.nan) for r in sh], float); xr = np.array([r.get(f, np.nan) for r in rest], float)
        ss, sr = np.nanstd(xs), np.nanstd(xr)
        ranges.append((f, ss / sr if sr > 0 else np.nan))
    ranges.sort(key=lambda t: t[1])
    for f, ratio in ranges:
        flag = "  <-- collapsed" if ratio < 0.5 else ""
        print(f"     {f:<14} range-ratio={ratio:.2f}{flag}")

    print("\n  2. per-feature corr with ΔG on SHORTS ALONE (is signal there, just masked in global fit?):")
    cors = []
    for f in feats:
        x = np.array([r.get(f, np.nan) for r in sh], float); m = ~np.isnan(x)
        if m.sum() > 6 and np.std(x[m]) > 0:
            cors.append((f, pearsonr(x[m], y[m])[0], spearmanr(x[m], y[m]).statistic))
    cors.sort(key=lambda t: -abs(t[1]))
    for f, rp, rs_ in cors[:8]:
        print(f"     {f:<14} Pearson={rp:+.3f}  Spearman={rs_:+.3f}")

    print("\n  3. short-ONLY LOO fit (can shorts predict themselves with best few feats?):")
    best = [t[0] for t in cors[:3]]

    def loo(rs, cols):
        X = np.array([[r[c] for c in cols] for r in rs], float); yy = np.array([r["y"] for r in rs])
        ok = ~np.isnan(X).any(1); X, yy, rs2 = X[ok], yy[ok], [rr for rr, o in zip(rs, ok) if o]
        pred = np.zeros(len(X))
        for i in range(len(X)):
            tr = np.arange(len(X)) != i
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            A = np.column_stack([np.ones(tr.sum()), (X[tr] - mu) / sd]); R = np.eye(A.shape[1]); R[0, 0] = 0
            w = np.linalg.solve(A.T @ A + R, A.T @ yy[tr]); pred[i] = np.r_[1.0, (X[i] - mu) / sd] @ w
        return pearsonr(pred, yy)[0] if len(X) > 5 else np.nan
    print(f"     short-only LOO with top3 {best}: r={loo(sh, best):+.3f}")
    print(f"     short-only LOO with full PROD:        r={loo(sh, PROD):+.3f}")

    print("\n  4. SEQUENCE-level signal shorts might need (composition the structure features miss):")
    HYD = set("AVLIMFWCY"); AROM = set("FWY")
    for r in sh:
        s = r["seq"]; L = max(1, len(s))
        r["s_hyd"] = sum(c in HYD for c in s) / L
        r["s_arom"] = sum(c in AROM for c in s) / L
        r["s_maxhyd"] = max([{"A":1.8,"V":4.2,"L":3.8,"I":4.5,"F":2.8,"W":-0.9,"M":1.9,"C":2.5,"Y":-1.3}.get(c,0) for c in s] + [0])
        r["s_bulky"] = sum(c in "FWYRK" for c in s) / L
    for f in ["s_hyd", "s_arom", "s_maxhyd", "s_bulky"]:
        x = np.array([r[f] for r in sh]); m = ~np.isnan(x)
        if np.std(x[m]) > 0:
            print(f"     seq {f:<10} corr ΔG = {pearsonr(x[m], y[m])[0]:+.3f}")


def main():
    rows, ns = load_rows()
    partA(rows, ns)
    partB(rows, ns)


if __name__ == "__main__":
    main()
