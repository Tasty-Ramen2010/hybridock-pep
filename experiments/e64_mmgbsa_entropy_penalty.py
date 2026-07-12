"""E64 — does an EXTENDEDNESS / conformational-entropy penalty fix MM-GBSA's cross-dataset flip?

Single-snapshot MM-GBSA omits −TΔS_conf. Hypothesis: it over-rates EXTENDED peptides (which pay an
entropy cost it can't see), so its error tracks rg_per_L. Fix: ΔG_corr = MMGBSA + α·extendedness.
Tests on cached e63_catalog (156 complexes, both datasets):
 1. raw MMGBSA flip + per-BSA / per-L normalization — does normalizing un-flip it?
 2. RESIDUAL diagnostic: is MMGBSA's error correlated with rg_per_L (= the missing entropy term)?
 3. PENALTY model: ΔG_corr = a·mmgbsa + b·rg_per_L + c — calibrate pooled, cross-dataset transfer.
 4. report α in kcal/mol per unit extendedness (interpretable penalty).
Also: rg_per_L added to the intensive feature model, refit + transfer.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

CAT = json.loads(Path("/tmp/e63_catalog.json").read_text())
rows = [r for r in CAT.values() if not (np.isnan(r.get("mmgbsa", np.nan)))]
cr = [r for r in rows if r["ds"] == "cr65"]
t98 = [r for r in rows if r["ds"] == "the98"]


def sp(rs, f, t="y"):
    a = np.array([r[f] for r in rs], float); b = np.array([r[t] for r in rs], float)
    m = ~(np.isnan(a) | np.isnan(b))
    return spearmanr(a[m], b[m]).statistic


def add_derived(r):
    r["mmgbsa_perBSA"] = r["mmgbsa"] / (r["total_bsa"] + 1e-6)
    r["mmgbsa_perL"] = r["mmgbsa"] / max(1, r["L"])
    return r


for r in rows:
    add_derived(r)


def transfer(cols):
    def fp(tr, te):
        X = np.array([[r[c] for c in cols] for r in tr], float); y = np.array([r["y"] for r in tr])
        ok = ~np.isnan(X).any(1); X, y = X[ok], y[ok]
        mu, sd = X.mean(0), X.std(0) + 1e-9
        A = np.column_stack([np.ones(len(X)), (X - mu) / sd]); R = 1.0 * np.eye(A.shape[1]); R[0, 0] = 0
        w = np.linalg.solve(A.T @ A + R, A.T @ y)
        Xe = np.array([[r[c] for c in cols] for r in te], float); oke = ~np.isnan(Xe).any(1)
        ye = np.array([r["y"] for r in te])[oke]
        return pearsonr(np.column_stack([np.ones(oke.sum()), (Xe[oke] - mu) / sd]) @ w, ye)[0]
    return fp(t98, cr), fp(cr, t98)


def main():
    print(f"=== E64 MM-GBSA entropy penalty.  cr65={len(cr)} the98={len(t98)} ===\n")

    print("=== (1) raw vs normalized MM-GBSA: corr(·,ΔG) per dataset (does normalizing un-flip?) ===")
    print(f"{'variant':<18}{'cr65':>9}{'the98':>9}  stable?   transfer 98→65 / 65→98")
    for f in ["mmgbsa", "mmgbsa_perBSA", "mmgbsa_perL"]:
        c, t = sp(cr, f), sp(t98, f)
        tr = transfer([f])
        st = "YES" if c * t > 0 else "flip"
        print(f"  {f:<16}{c:>+9.3f}{t:>+9.3f}  {st:<6}   {tr[0]:+.3f} / {tr[1]:+.3f}")

    print("\n=== (2) RESIDUAL diagnostic: is MM-GBSA's error the missing entropy term? ===")
    # global fit mmgbsa->y, residual vs rg_per_L per dataset
    g = np.array([r["mmgbsa"] for r in rows]); y = np.array([r["y"] for r in rows])
    a, b = np.polyfit(g, y, 1);
    for r in rows:
        r["resid"] = r["y"] - (a * r["mmgbsa"] + b)
    print("  corr(MMGBSA residual, rg_per_L):")
    for nm, rs in [("cr65", cr), ("the98", t98), ("POOLED", rows)]:
        print(f"    {nm:<8} Spearman={sp(rs,'rg_per_L','resid'):+.3f}")
    print("  (consistent NEGATIVE => extended peptides: MMGBSA says too strong, real is weaker = entropy gap)")

    print("\n=== (3) PENALTY MODEL: ΔG_corr = a·mmgbsa + b·rg_per_L + c (calibrate pooled, transfer) ===")
    print(f"{'model':<26}{'cr65→98':>10}{'98→cr65':>10}")
    for nm, cols in [("mmgbsa (baseline)", ["mmgbsa"]),
                     ("mmgbsa + rg_per_L", ["mmgbsa", "rg_per_L"]),
                     ("mmgbsa + rg_per_L + hyd", ["mmgbsa", "rg_per_L", "hyd_frac"]),
                     ("mmgbsa_perBSA + rg_per_L", ["mmgbsa_perBSA", "rg_per_L"])]:
        t1, t2 = transfer(cols)
        print(f"  {nm:<24}{t2:>+10.3f}{t1:>+10.3f}")

    # interpretable alpha: pooled OLS y ~ mmgbsa + rg_per_L
    X = np.array([[r["mmgbsa"], r["rg_per_L"]] for r in rows]); Y = np.array([r["y"] for r in rows])
    A = np.column_stack([np.ones(len(X)), X])
    coef = np.linalg.lstsq(A, Y, rcond=None)[0]
    print(f"\n  pooled fit: ΔG = {coef[0]:+.2f} {coef[1]:+.4f}·mmgbsa {coef[2]:+.2f}·rg_per_L")
    print(f"  => extendedness penalty α = {coef[2]:+.2f} kcal/mol per unit rg_per_L "
          f"(rg_per_L spans ~{np.ptp([r['rg_per_L'] for r in rows]):.2f} => up to "
          f"{abs(coef[2])*np.ptp([r['rg_per_L'] for r in rows]):.1f} kcal/mol swing)")
    pooled_pred = A @ coef
    print(f"  pooled Pearson: mmgbsa-only={pearsonr([r['mmgbsa'] for r in rows],Y)[0]:+.3f} "
          f"-> +penalty={pearsonr(pooled_pred,Y)[0]:+.3f}")

    print("\n=== (4) rg_per_L into the intensive feature model — refit + transfer ===")
    print(f"{'feature set':<34}{'cr65→98':>10}{'98→cr65':>10}")
    for nm, cols in [("intensive base (burial+hyd)", ["mean_burial", "hyd_frac"]),
                     ("+ rg_per_L", ["mean_burial", "hyd_frac", "rg_per_L"]),
                     ("+ rg_per_L + e2e_per_L", ["mean_burial", "hyd_frac", "rg_per_L", "e2e_per_L"]),
                     ("mmgbsa+rg_per_L+burial+hyd", ["mmgbsa", "rg_per_L", "mean_burial", "hyd_frac"])]:
        t1, t2 = transfer(cols)
        flag = "  <== best" if min(t1, t2) > 0.15 else ""
        print(f"  {nm:<32}{t2:>+10.3f}{t1:>+10.3f}{flag}")


if __name__ == "__main__":
    main()
