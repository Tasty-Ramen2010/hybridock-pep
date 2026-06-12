"""M1b — WHY doesn't the residual ML transfer? Full diagnosis + Ram's hypotheses.

Tests, in order:
 1. SIGN-FLIP diagnostic: corr(feature, residual) in crystal-65 vs the-98 SEPARATELY. A feature
    whose sign flips across datasets is transfer-POISON — the ML learns one sign, applies it
    backwards. This is the single most important "why" check.
 2. Per-COMPLEX failure table: pred vs actual, which structures we miss and by how much, keyed to
    L, SASA, hydrophobic/hydrophilic, charge.
 3. (b) minimal feature sets: net_charge only, transferable-only vs all-11.
 4. Nonlinearity (Ram): log(L), interaction terms (charge×burial, charged×L), hydrophobic/hydrophilic
    ratio; and a shallow gradient-boosted model vs linear ridge — cross-dataset.
 5. Richer physics: total/polar/hydrophobic buried SASA (were we missing the hydrophilic side?).

All metrics leave-DATASET-out, Spearman (robust), charge-stratified.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hybridock_pep.scoring.geometry_features import compute_geometry_features  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

A3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
      "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
      "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
HYD = set("AVLIMFWC"); HPHIL = set("RNDQEKHSTY")


def feats(seq, g, ei, ei_std, mts, L):
    nq = seq.count("K") + seq.count("R") - seq.count("D") - seq.count("E")
    cf = sum(c in "DEKR" for c in seq) / max(1, L)
    hyd = sum(c in HYD for c in seq) / max(1, L)
    phil = sum(c in HPHIL for c in seq) / max(1, L)
    bsa_hyd = g.get("bsa_hyd", 0.0); sasa_hb = g.get("sasa_hb", 0.0); sasa_sb = g.get("sasa_sb", 0.0)
    bsa_tot = bsa_hyd + sasa_hb + sasa_sb + 1e-6
    return dict(
        # linear
        net_charge=nq, abs_charge=abs(nq), charged_frac=cf, hyd_frac=hyd, phil_frac=phil,
        L=L, bsa_hyd=bsa_hyd, sasa_hb=sasa_hb, sasa_sb=sasa_sb, mj=g.get("mj_contact", 0.0),
        strength=g.get("strength_bur", 0.0), e_int_std=ei_std, mts=mts,
        e_int_perL=ei / max(1, L),
        # Ram's nonlinear / interaction / ratio
        logL=np.log(L), hyd_over_phil=hyd / (phil + 1e-3),
        bsa_hyd_frac=bsa_hyd / bsa_tot, bsa_polar_frac=(sasa_hb + sasa_sb) / bsa_tot,
        charge_x_bsa=nq * bsa_hyd, charged_x_L=cf * L, absq_x_phil=abs(nq) * phil,
        charged_frac_sq=cf * cf,
    )


def build(which):
    if which == "cr":
        bench = {r["pdb"].upper(): r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
        e = json.loads(Path("/tmp/e49_ens_mmgbsa.json").read_text())
        rows = []
        for k, m in bench.items():
            if k not in e or not m.get("peptide_seq"):
                continue
            pose = ROOT / f"logs/crystal65_n100/cr_{k}/poses/pose_0.pdb"; rec = ROOT / m["pocket_pdb"]
            ei = e[k]["e_int_mean"]
            if not pose.exists() or not np.isfinite(ei) or abs(ei) > 1e8:
                continue
            g = compute_geometry_features(pose, rec) or {}
            seq = m["peptide_seq"]
            rows.append(dict(feats(seq, g, ei, e[k]["e_int_std"], e[k]["minus_tds"], len(seq)),
                             pdb=k, y=m["dg_exp"], cf=e[k]["cf"], seq=seq))
        return rows
    e = json.loads(Path("/tmp/e49b_the98.json").read_text()); work = Path("/tmp/ppep_work")
    rows = []
    for k, v in e.items():
        ei = v["e_int_mean"]
        if not np.isfinite(ei) or abs(ei) > 1e8:
            continue
        g = compute_geometry_features(work / f"{k}_pep.pdb", work / f"{k}_rec.pdb") or {}
        seq = v["seq"]
        rows.append(dict(feats(seq, g, ei, v["e_int_std"], v["minus_tds"], len(seq)),
                         pdb=k, y=v["y"], cf=v["cf"], seq=seq))
    return rows


def _resid(rows):
    """baseline winsorized e_int_perL -> y; return residual array + baseline preds."""
    x = np.array([r["e_int_perL"] for r in rows]); y = np.array([r["y"] for r in rows])
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    a, b = np.polyfit(np.clip(x, lo, hi), y, 1)
    base = a * np.clip(x, lo, hi) + b
    return y - base, base, (a, b, lo, hi)


ALL = ["net_charge", "charged_frac", "hyd_frac", "phil_frac", "L", "bsa_hyd", "sasa_hb",
       "sasa_sb", "mj", "strength", "e_int_std", "mts"]
NONLIN = ["net_charge", "logL", "hyd_over_phil", "bsa_hyd_frac", "bsa_polar_frac",
          "charge_x_bsa", "charged_x_L", "absq_x_phil", "charged_frac_sq", "strength"]


def signflip(cr, b98):
    rc, _, _ = _resid(cr); rb, _, _ = _resid(b98)
    print("=== 1. SIGN-FLIP diagnostic: corr(feature, residual) per dataset ===")
    print(f"  {'feature':<16}{'crystal-65':>11}{'the-98':>9}{'verdict':>14}")
    for f in ALL + ["logL", "hyd_over_phil", "charge_x_bsa", "bsa_polar_frac"]:
        vc = np.array([r[f] for r in cr]); vb = np.array([r[f] for r in b98])
        a = pearsonr(vc, rc).statistic if vc.std() > 0 else 0
        b = pearsonr(vb, rb).statistic if vb.std() > 0 else 0
        v = "TRANSFERS" if a * b > 0.01 else ("FLIPS (poison)" if a * b < -0.01 else "weak/none")
        print(f"  {f:<16}{a:>+11.3f}{b:>+9.3f}{v:>14}")


def failtable(cr, b98):
    # train crystal -> predict the-98, ridge all feats, show worst misses with structure props
    rc, _, cal = _resid(cr)
    Xtr = np.array([[r[f] for f in ALL] for r in cr])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
    R = 10 * np.eye(A.shape[1]); R[0, 0] = 0
    w = np.linalg.solve(A.T @ A + R, A.T @ rc)
    a, b, lo, hi = cal
    print("\n=== 2. Per-COMPLEX failures (train crystal-65 -> test the-98), 12 worst ===")
    print(f"  {'pdb':<12}{'y':>6}{'pred':>7}{'err':>6}{'L':>3}{'cf':>5}{'nq':>4}{'hyd':>5}{'bsaH':>6}  seq")
    out = []
    for r in b98:
        base = a * np.clip(r["e_int_perL"], lo, hi) + b
        ml = np.r_[1, (np.array([r[f] for f in ALL]) - mu) / sd] @ w
        pred = base + ml
        out.append((abs(pred - r["y"]), r, pred, pred - r["y"]))
    for err, r, pred, signed in sorted(out, key=lambda t: -t[0])[:12]:
        print(f"  {r['pdb']:<12}{r['y']:>6.1f}{pred:>7.1f}{signed:>+6.1f}{r['L']:>3}{r['cf']:>5.2f}"
              f"{r['net_charge']:>4}{r['hyd_frac']:>5.2f}{r['bsa_hyd']:>6.1f}  {r['seq'][:14]}")


def _cross(trn, ten, fl, model="ridge"):
    rtr, _, cal = _resid(trn)
    a, b, lo, hi = cal
    Xtr = np.array([[r[f] for f in fl] for r in trn]); Xte = np.array([[r[f] for f in fl] for r in ten])
    base_te = np.array([a * np.clip(r["e_int_perL"], lo, hi) + b for r in ten])
    if model == "ridge":
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        A = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
        R = 10 * np.eye(A.shape[1]); R[0, 0] = 0
        w = np.linalg.solve(A.T @ A + R, A.T @ rtr)
        ml = np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd]) @ w
    else:  # shallow gradient boosting
        from sklearn.ensemble import GradientBoostingRegressor
        m = GradientBoostingRegressor(n_estimators=60, max_depth=2, learning_rate=0.05,
                                      subsample=0.8, random_state=0)
        m.fit(Xtr, rtr); ml = m.predict(Xte)
    full = base_te + ml
    y = np.array([r["y"] for r in ten]); cf = np.array([r["cf"] for r in ten]); h = cf >= 0.3
    return (spearmanr(base_te, y).statistic, spearmanr(base_te[h], y[h]).statistic,
            spearmanr(full, y).statistic, spearmanr(full[h], y[h]).statistic)


def subsets(cr, b98):
    print("\n=== 3+4. feature sets x model, cross-dataset Spearman (base->+ML, all / charged) ===")
    sets = {"net_charge ONLY (b)": ["net_charge"],
            "3 minimal": ["net_charge", "bsa_hyd", "strength"],
            "all-12 linear": ALL,
            "nonlinear+interactions": NONLIN}
    for nm, fl in sets.items():
        for mdl in ("ridge", "gbt"):
            try:
                r = _cross(cr, b98, fl, mdl)
                r2 = _cross(b98, cr, fl, mdl)
                print(f"  {nm:<24}[{mdl:>5}]  cr->98 base/ML all {r[0]:+.2f}/{r[2]:+.2f} chg {r[1]:+.2f}/{r[3]:+.2f}"
                      f"  | 98->cr chg {r2[1]:+.2f}/{r2[3]:+.2f}")
            except Exception as e:  # noqa: BLE001
                print(f"  {nm:<24}[{mdl:>5}]  ERR {str(e)[:30]}")


def main():
    cr = build("cr"); b98 = build("b98")
    print(f"crystal-65 n={len(cr)} | the-98 n={len(b98)}\n")
    signflip(cr, b98)
    failtable(cr, b98)
    subsets(cr, b98)
    print("\n  >> if features FLIP across datasets, NO model (linear/log/GBT) transfers them = the wall.")
    print("  >> if a nonlinear/GBT set beats ridge AND beats baseline on charged BOTH directions = real.")


if __name__ == "__main__":
    main()
