"""E67 — anatomy of strong vs weak binders. What ARE they, can we rank them, what do we miss?

Ram: "something doesn't magically become strong — what are they?" Decompose:
 (A) characterize strong (most −ΔG tercile) vs weak (least) WITHIN each dataset (pooling is confounded
     with dataset identity). Cohen's d per feature; flag sign-CONSISTENT discriminators (real) vs flips.
 (B) can our corrected predictor (mmgbsa+rg_per_L) RANK strong vs weak? per-dataset Spearman + how many
     true-strong land in predicted-strong half.
 (C) the MISSES: true-strong we rank low — what feature do they share that we don't capture? -> the more
     fundamental characteristic.
 (D) hydrophobic term test: does a hyd_frac reward/penalty help, what SIGN, how many kcal/mol?
 (E) pocket vs peptide (the-98 has poc_* in e28): is strength in the INTERFACE not the peptide?
Data: /tmp/e63_catalog.json (+ /tmp/e28_feats.json for pocket).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

CAT = json.loads(Path("/tmp/e63_catalog.json").read_text())
E28 = json.loads(Path("/tmp/e28_feats.json").read_text())
rows = [r for r in CAT.values() if not np.isnan(r.get("mmgbsa", np.nan))]
cr = [r for r in rows if r["ds"] == "cr65"]
t98 = [r for r in rows if r["ds"] == "the98"]

FEATS = ["total_bsa", "mean_burial", "max_burial", "n_anchor", "nonbind_frac", "dangling_frac",
         "buried_frac", "rg_per_L", "e2e_per_L", "L", "charged_frac", "net_charge", "abs_net_charge",
         "hyd_frac", "arom_frac", "bulky_frac", "pro_frac", "gly_frac", "polar_frac", "gravy",
         "mmgbsa", "eint", "mtds"]


def cohen_d(strong, weak, f):
    a = np.array([r[f] for r in strong], float); b = np.array([r[f] for r in weak], float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    sp = np.sqrt((a.var() + b.var()) / 2) + 1e-9
    return (a.mean() - b.mean()) / sp


def terciles(ds):
    s = sorted(ds, key=lambda r: r["y"])
    k = len(s) // 3
    return s[:k], s[-k:]  # strong (most -ve), weak (least -ve)


def main():
    crs, crw = terciles(cr)
    ts, tw = terciles(t98)
    print(f"=== (A) WHAT ARE strong vs weak? Cohen's d (strong−weak), within each dataset ===")
    print(f"  strong = most-negative ΔG tercile; cr65 |{len(crs)}| the98 |{len(ts)}|")
    print(f"{'feature':<14}{'cr65 d':>9}{'the98 d':>9}   consistent discriminator?")
    res = []
    for f in FEATS:
        dc = cohen_d(crs, crw, f); dt = cohen_d(ts, tw, f)
        res.append((f, dc, dt))
    for f, dc, dt in sorted(res, key=lambda t: -(abs(t[1]) + abs(t[2])) / 2):
        cons = "YES" if dc * dt > 0 and min(abs(dc), abs(dt)) > 0.3 else ("flip" if dc * dt < 0 else "")
        mark = "  <== REAL (both dirs)" if cons == "YES" else ("  (dataset-specific/flips)" if cons == "flip" else "")
        print(f"  {f:<12}{dc:>+9.2f}{dt:>+9.2f}   {cons}{mark}")

    print("\n=== (B) CAN WE RANK strong vs weak? corrected pred = mmgbsa + rg_per_L (within dataset) ===")
    for nm, ds in [("cr65", cr), ("the98", t98)]:
        X = np.array([[r["mmgbsa"], r["rg_per_L"]] for r in ds]); y = np.array([r["y"] for r in ds])
        A = np.column_stack([np.ones(len(X)), X]); w = np.linalg.lstsq(A, y, rcond=None)[0]
        pred = A @ w
        # of true-strong tercile, how many in predicted-strong half?
        st, wk = terciles(ds)
        thr = np.median(pred)
        true_strong = set(id(r) for r in st)
        hit = sum(1 for i, r in enumerate(ds) if id(r) in true_strong and pred[i] < thr)
        print(f"  {nm:<6} Spearman(pred,exp)={spearmanr(pred,y).statistic:+.3f}  "
              f"true-strong caught in predicted-strong half: {hit}/{len(st)}")

    print("\n=== (C) THE MISSES: true-strong we rank LOW — what do they share? (the98, n largest) ===")
    X = np.array([[r["mmgbsa"], r["rg_per_L"]] for r in t98]); y = np.array([r["y"] for r in t98])
    A = np.column_stack([np.ones(len(X)), X]); w = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = A @ w; resid = y - pred  # very negative resid = much stronger than we predict = a MISS
    st, _ = terciles(t98)
    miss = sorted(st, key=lambda r: y[t98.index(r)] - pred[t98.index(r)])[:6]  # strongest under-pred
    caught = sorted(st, key=lambda r: -(y[t98.index(r)] - pred[t98.index(r)]))[:6]
    print("  MISSED strong (we under-rate most):")
    for r in miss:
        i = t98.index(r)
        print(f"   exp={r['y']:+.1f} pred={pred[i]:+.1f} | hyd={r['hyd_frac']:.2f} arom={r['arom_frac']:.2f} "
              f"chg={r['net_charge']:+d} bsa={r['total_bsa']:.0f} burDen={r['mean_burial']:.0f} {r['seq'][:16]}")
    print("  CAUGHT strong (we rate well):")
    for r in caught:
        i = t98.index(r)
        print(f"   exp={r['y']:+.1f} pred={pred[i]:+.1f} | hyd={r['hyd_frac']:.2f} arom={r['arom_frac']:.2f} "
              f"chg={r['net_charge']:+d} bsa={r['total_bsa']:.0f} burDen={r['mean_burial']:.0f} {r['seq'][:16]}")
    # quantify: feature means missed vs caught
    print("  feature contrast (missed − caught):")
    for f in ["hyd_frac", "arom_frac", "abs_net_charge", "charged_frac", "total_bsa", "mean_burial", "max_burial", "polar_frac"]:
        dm = np.mean([r[f] for r in miss]) - np.mean([r[f] for r in caught])
        print(f"    {f:<14} {dm:+.3f}")

    print("\n=== (D) HYDROPHOBIC term: does it help, what SIGN/magnitude? ===")
    def transfer(cols):
        def fp(tr, te):
            X = np.array([[r[c] for c in cols] for r in tr], float); y = np.array([r["y"] for r in tr])
            mu, sd = X.mean(0), X.std(0) + 1e-9
            A = np.column_stack([np.ones(len(X)), (X - mu) / sd]); R = np.eye(A.shape[1]); R[0, 0] = 0
            w = np.linalg.solve(A.T @ A + R, A.T @ y)
            Xe = np.array([[r[c] for c in cols] for r in te], float)
            return pearsonr(np.column_stack([np.ones(len(Xe)), (Xe - mu) / sd]) @ w, [r["y"] for r in te])[0]
        return fp(t98, cr), fp(cr, t98)
    for nm, cols in [("mmgbsa+rg_per_L", ["mmgbsa", "rg_per_L"]),
                     ("+ hyd_frac", ["mmgbsa", "rg_per_L", "hyd_frac"]),
                     ("+ bulky_frac", ["mmgbsa", "rg_per_L", "bulky_frac"]),
                     ("+ arom_frac", ["mmgbsa", "rg_per_L", "arom_frac"])]:
        t1, t2 = transfer(cols)
        print(f"  {nm:<20} transfer 98→65={t1:+.3f}  65→98={t2:+.3f}")
    # calibrated hyd coefficient (pooled OLS, interpretable kcal/mol)
    X = np.array([[r["mmgbsa"], r["rg_per_L"], r["hyd_frac"]] for r in rows]); Y = np.array([r["y"] for r in rows])
    A = np.column_stack([np.ones(len(X)), X]); c = np.linalg.lstsq(A, Y, rcond=None)[0]
    print(f"  pooled fit: hyd_frac coef = {c[3]:+.2f} kcal/mol per unit frac "
          f"({'REWARD (more hydrophobic->stronger)' if c[3] < 0 else 'penalty'}); "
          f"full hyd swing ~{abs(c[3])*0.5:.1f} kcal/mol")

    print("\n=== (E) POCKET vs PEPTIDE: is strength in the INTERFACE? (the98, poc_* from e28) ===")
    sub = [r for r in t98 if f"{r['ds']}"]
    pk = {k: E28[k.split('_', 1)[1]] for k in [f"98_{kk}" for kk in []] }  # placeholder
    # map e28 by pdb key (e28 keys look like '1AQC_B_D')
    pf = []
    for r in t98:
        # e63 key was '98_<pdb_B_D>'; recover by matching seq? store via CAT keys
        pass
    # rebuild from CAT keys to get e28 join
    joined = []
    for k, r in CAT.items():
        if r["ds"] != "the98":
            continue
        e = E28.get(k[3:])  # strip '98_'
        if e:
            joined.append({**r, **{f"poc_{x}": e.get(f"poc_{x}", np.nan) for x in ["n", "f_hyd", "f_arom", "net", "eis"]}})
    if joined:
        ys = [r["y"] for r in joined]
        for f in ["poc_f_hyd", "poc_f_arom", "poc_eis", "poc_n", "poc_net"]:
            xs = [r.get(f, np.nan) for r in joined]
            print(f"  corr({f:<10}, ΔG) the98 = {spearmanr(xs, ys).statistic:+.3f}")
        # complementarity: peptide hyd × pocket hyd
        comp = [r["hyd_frac"] * r.get("poc_f_hyd", 0) for r in joined]
        print(f"  corr(pep_hyd × pocket_hyd COMPLEMENTARITY, ΔG) = {spearmanr(comp, ys).statistic:+.3f}")
    print("\n  >> if pocket/complementarity beats peptide-only features, strength is INTERFACE not peptide.")


if __name__ == "__main__":
    main()
