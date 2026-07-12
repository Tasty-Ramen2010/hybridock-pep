"""E73 — can vdW/packing + shape/hydrophobicity separate strong vs weak? (Ram's redirect.)

E72 showed vdw separates CHARGED binders at +0.50 (packing, not charge). Test it properly:
 (A) vdw as a strength separator: full / charged / low-Q, raw and intensive (per-L, per-BSA = packing
     QUALITY vs quantity), with cross-dataset transfer (the honest test).
 (B) per-binder feature analysis: vdw-predicted vs experimental, who is mis-ranked, what they share.
 (C) shape/hydrophobicity hypotheses to differentiate strong/weak:
       H1 vdw/BSA            packing density (well-filled pocket)
       H2 vdw + hyd_frac     hydrophobic reward on top of packing
       H3 vdw + rg_per_L     packing + compactness (penalize extended)
       H4 hyd complementarity pep_hyd × pocket_hyd (interface match)
       H5 vdw × hyd_frac     hydrophobic packing (shape AND chemistry aligned)
       H6 vdw/mean_burial    packing per buried residue
Join: e72 (vdw/coul/gbpol) + e63 catalog (rg_per_L, hyd_frac, bsa, burial, pocket).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parents[1]
E72 = json.loads(Path("/tmp/e72_elec.json").read_text())
E63 = json.loads(Path("/tmp/e63_catalog.json").read_text())
E28 = json.loads(Path("/tmp/e28_feats.json").read_text())


def join():
    rows = []
    for k, e in E72.items():
        if not (np.isfinite(e["vdw"]) and abs(e["vdw"]) < 1e5):
            continue
        c = E63.get(f"98_{k}")
        if c is None:
            continue
        p = E28.get(k, {})
        L = max(1, e["L"])
        bsa = c.get("total_bsa", c.get("bsa_hyd", np.nan))
        r = dict(
            id=k, y=e["y"], net_charge=e["net_charge"], L=L,
            vdw=e["vdw"], coul=e["coul"], gbpol=e["gbpol"],
            vdw_L=e["vdw"] / L,
            vdw_bsa=e["vdw"] / (bsa + 1e-6) if np.isfinite(bsa) else np.nan,
            vdw_bur=e["vdw"] / (c.get("mean_burial", 50) + 1e-6),
            rg_per_L=c.get("rg_per_L", np.nan), hyd_frac=c.get("hyd_frac", np.nan),
            bsa=bsa, mean_burial=c.get("mean_burial", np.nan),
            poc_f_hyd=p.get("poc_f_hyd", np.nan),
        )
        r["hyd_compl"] = r["hyd_frac"] * r["poc_f_hyd"] if np.isfinite(r["poc_f_hyd"]) else np.nan
        r["vdw_x_hyd"] = r["vdw"] * r["hyd_frac"]
        rows.append(r)
    return rows


def sp(rows, f):
    x = np.array([r[f] for r in rows], float); y = np.array([r["y"] for r in rows], float)
    m = ~(np.isnan(x) | np.isnan(y))
    return spearmanr(x[m], y[m]).statistic if m.sum() > 5 else np.nan


def transfer(rows, cols):
    cr = [r for r in rows if r["id"]]  # all the-98 here; split by charge instead for transfer proxy
    # honest cross-split: train low-Q -> test charged, and vice versa (different regimes)
    ch = [r for r in rows if abs(r["net_charge"]) >= 2]
    lo = [r for r in rows if abs(r["net_charge"]) < 2]

    def fp(tr, te):
        X = np.array([[r[c] for c in cols] for r in tr], float); y = np.array([r["y"] for r in tr])
        ok = ~np.isnan(X).any(1); X, y = X[ok], y[ok]
        if len(X) < 8:
            return np.nan
        mu, sd = X.mean(0), X.std(0) + 1e-9
        A = np.column_stack([np.ones(len(X)), (X - mu) / sd]); R = np.eye(A.shape[1]); R[0, 0] = 0
        w = np.linalg.solve(A.T @ A + R, A.T @ y)
        Xe = np.array([[r[c] for c in cols] for r in te], float); oke = ~np.isnan(Xe).any(1)
        if oke.sum() < 5:
            return np.nan
        return pearsonr(np.column_stack([np.ones(oke.sum()), (Xe[oke] - mu) / sd]) @ w,
                        np.array([r["y"] for r in te])[oke])[0]
    return fp(lo, ch), fp(ch, lo)


def main():
    rows = join()
    ch = [r for r in rows if abs(r["net_charge"]) >= 2]
    lo = [r for r in rows if abs(r["net_charge"]) < 2]
    print(f"=== E73 vdW/packing + shape strength separation. n={len(rows)} "
          f"(charged={len(ch)}, low-Q={len(lo)}) ===")

    print("\n=== (A) vdW as strength separator (Spearman vs ΔG) ===")
    print(f"{'feature':<14}{'ALL':>8}{'charged':>9}{'low-Q':>8}   meaning")
    for f, desc in [("vdw", "raw packing"), ("vdw_L", "packing/residue"),
                    ("vdw_bsa", "packing density (QUALITY)"), ("vdw_bur", "packing/burial")]:
        print(f"  {f:<12}{sp(rows,f):>+8.3f}{sp(ch,f):>+9.3f}{sp(lo,f):>+8.3f}   {desc}")

    print("\n=== (B) per-binder analysis: vdw-rank misses (charged subset) ===")
    y = np.array([r["y"] for r in ch]); v = np.array([r["vdw"] for r in ch])
    a, b = np.polyfit(v, y, 1); pred = a * v + b; resid = y - pred
    order = np.argsort(resid)
    print("  vdw OVER-rates (predicts too strong):")
    for i in order[:4]:
        r = ch[i]
        print(f"   {r['id']:<12} exp={r['y']:+.1f} vdwPred={pred[i]:+.1f} | hyd={r['hyd_frac']:.2f} "
              f"rgL={r['rg_per_L']:.2f} bsa={r['bsa']:.0f} Q={r['net_charge']:+d}")
    print("  vdw UNDER-rates (predicts too weak):")
    for i in order[::-1][:4]:
        r = ch[i]
        print(f"   {r['id']:<12} exp={r['y']:+.1f} vdwPred={pred[i]:+.1f} | hyd={r['hyd_frac']:.2f} "
              f"rgL={r['rg_per_L']:.2f} bsa={r['bsa']:.0f} Q={r['net_charge']:+d}")

    print("\n=== (C) shape/hydrophobicity hypotheses (within-charged Spearman + cross-regime transfer) ===")
    print(f"{'hypothesis':<26}{'charged r':>11}{'lo→ch':>8}{'ch→lo':>8}")
    H = {
        "vdw [baseline]": ["vdw"],
        "H1 vdw/BSA (density)": ["vdw_bsa"],
        "H2 vdw + hyd_frac": ["vdw", "hyd_frac"],
        "H3 vdw + rg_per_L": ["vdw", "rg_per_L"],
        "H4 hyd complementarity": ["hyd_compl"],
        "H5 vdw × hyd_frac": ["vdw_x_hyd"],
        "H6 vdw/mean_burial": ["vdw_bur"],
        "H7 vdw+hyd+rg (combined)": ["vdw", "hyd_frac", "rg_per_L"],
    }
    for nm, cols in H.items():
        # within-charged: multi-feature LOO-ish single fit Spearman of fitted pred
        sub = [r for r in ch if not any(np.isnan(r[c]) for c in cols)]
        if len(sub) >= 8:
            X = np.array([[r[c] for c in cols] for r in sub]); yy = np.array([r["y"] for r in sub])
            mu, sd = X.mean(0), X.std(0) + 1e-9
            A = np.column_stack([np.ones(len(X)), (X - mu) / sd])
            w = np.linalg.lstsq(A, yy, rcond=None)[0]
            rch = spearmanr(A @ w, yy).statistic
        else:
            rch = np.nan
        t1, t2 = transfer(rows, cols)
        flag = "  <== best" if (not np.isnan(rch) and rch < -0.4) else ""
        print(f"  {nm:<26}{rch:>+11.3f}{t1:>+8.3f}{t2:>+8.3f}{flag}")
    print("\n  (charged r: more negative = better separates strong; transfer lo↔ch = does it generalize)")


if __name__ == "__main__":
    main()
