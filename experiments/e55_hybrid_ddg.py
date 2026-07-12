"""E55 — beat FlexPepDock: hybrid ΔΔG (ref2015 + MM-GBSA + ensemble + mutation physics).

FlexPepDock = ref2015 (1PPF Spearman 0.56). MM-GBSA adds GB desolvation (orthogonal on CHARGED muts
where ref2015 elec is crude); ensemble adds fluctuation. Hypothesis: a leave-COMPLEX-out hybrid beats
ref2015 alone on the SKEMPI ΔΔG benchmark. Includes LOSS-FEATURE analysis: where does ref2015 fail
(charged? buried? size?), and does MM-GBSA fix exactly those? Drives the next feature.

Predictors per mutation: ref2015 (e54), mmgbsa (e51), ensemble <E_int> (e52). Mutation physics:
Δhydrophobicity, Δvolume, Δcharge, |Δcharge|, pro/gly involved.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
CXS = ["1PPF_E_I", "1CHO_EFG_I", "1R0R_E_I", "3SGB_E_I", "1AO7_ABC_DE"]
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
      "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
      "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
VOL = {"A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8, "E": 138.4,
       "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9,
       "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0}
CHG = {"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.5}


def mutfeat(wt, mut):
    return dict(d_hyd=KD[mut] - KD[wt], d_vol=VOL[mut] - VOL[wt],
               d_chg=CHG.get(mut, 0) - CHG.get(wt, 0),
               abs_dchg=abs(CHG.get(mut, 0) - CHG.get(wt, 0)),
               charged_involved=1.0 if (wt in "DEKRH" or mut in "DEKRH") else 0.0,
               progly=1.0 if (mut in "PG" or wt in "PG") else 0.0)


def assemble():
    rows = []
    for cx in CXS:
        e51 = Path(f"/tmp/e51_{cx}.json"); e52 = Path(f"/tmp/e52_{cx}.json"); e54 = Path(f"/tmp/e54_{cx}.json")
        if not (e51.exists() and e54.exists()):
            continue
        d51 = json.loads(e51.read_text()); d54 = json.loads(e54.read_text())
        d52 = json.loads(e52.read_text()) if e52.exists() else {}
        wt54 = d54.get("WT")
        for k, v in d51.items():
            if k == "WT" or k not in d54:
                continue
            wt, mut = k[0], k[-1]
            if wt not in KD or mut not in KD:
                continue
            r = dict(cx=cx, mut=k, ddg_exp=v["ddg_exp"], ref2015=d54[k] - wt54,
                     mmgbsa=v["ddg_pred"], ens=d52.get(k, {}).get("ddg_ens", np.nan), **mutfeat(wt, mut))
            if abs(r["ref2015"]) < 50 and abs(r["mmgbsa"]) < 50:
                rows.append(r)
    return rows


def corr(p, e):
    from scipy.stats import pearsonr, spearmanr
    p, e = np.asarray(p), np.asarray(e)
    return pearsonr(p, e).statistic, spearmanr(p, e).statistic


def main():
    rows = assemble()
    from scipy.stats import pearsonr, spearmanr
    print(f"=== E55 hybrid ΔΔG — beat FlexPepDock (ref2015). n={len(rows)} ===\n")

    print("=== per-complex: each predictor vs experimental (Spearman) ===")
    print(f"  {'complex':<13}{'n':>4}{'ref2015':>9}{'mmgbsa':>8}{'ensemble':>9}")
    percx = {}
    for cx in CXS:
        sub = [r for r in rows if r["cx"] == cx]
        if len(sub) < 5:
            continue
        e = [r["ddg_exp"] for r in sub]
        rr = spearmanr([r["ref2015"] for r in sub], e).statistic
        rm = spearmanr([r["mmgbsa"] for r in sub], e).statistic
        ev = [r["ens"] for r in sub]
        re_ = spearmanr(ev, e).statistic if not np.isnan(ev).all() else float("nan")
        percx[cx] = (rr, rm, re_)
        print(f"  {cx:<13}{len(sub):>4}{rr:>+9.3f}{rm:>+8.3f}{re_:>+9.3f}")
    mref = np.nanmean([v[0] for v in percx.values()])
    print(f"  {'MEAN':<13}{'':>4}{mref:>+9.3f}{np.nanmean([v[1] for v in percx.values()]):>+8.3f}"
          f"{np.nanmean([v[2] for v in percx.values()]):>+9.3f}   <- FlexPepDock = ref2015 mean")

    print("\n=== LOSS ANALYSIS: where does ref2015 FAIL, and does MM-GBSA fix it? ===")
    # ref2015 residual per mutation (within-complex z to be fair)
    for r in rows:
        pass
    for grp, mask in [("charged-involved", lambda r: r["charged_involved"] > 0),
                      ("uncharged", lambda r: r["charged_involved"] == 0),
                      ("big Δvolume |Δv|>50", lambda r: abs(r["d_vol"]) > 50),
                      ("pro/gly", lambda r: r["progly"] > 0)]:
        sub = [r for r in rows if mask(r)]
        if len(sub) < 8:
            continue
        e = [r["ddg_exp"] for r in sub]
        rr = spearmanr([r["ref2015"] for r in sub], e).statistic
        rm = spearmanr([r["mmgbsa"] for r in sub], e).statistic
        print(f"  {grp:<20} n={len(sub):>3}  ref2015={rr:+.3f}  mmgbsa={rm:+.3f}  "
              f"{'<< MMGBSA helps' if rm > rr + 0.05 else ''}")

    print("\n=== HYBRID: leave-COMPLEX-out — does combining beat ref2015 alone? ===")
    feats_sets = {
        "ref2015 only [baseline]": ["ref2015"],
        "ref2015+mmgbsa": ["ref2015", "mmgbsa"],
        "ref2015+mmgbsa+ens": ["ref2015", "mmgbsa", "ens"],
        "ref2015+mmgbsa+Δphys": ["ref2015", "mmgbsa", "d_hyd", "d_vol", "abs_dchg"],
        "all": ["ref2015", "mmgbsa", "ens", "d_hyd", "d_vol", "d_chg", "abs_dchg", "charged_involved", "progly"],
    }
    for nm, fs in feats_sets.items():
        percx_h = []
        for cx in CXS:
            tr = [r for r in rows if r["cx"] != cx and not any(np.isnan(r[f]) for f in fs)]
            te = [r for r in rows if r["cx"] == cx and not any(np.isnan(r[f]) for f in fs)]
            if len(te) < 5 or len(tr) < 20:
                continue
            X = np.array([[r[f] for f in fs] for r in tr]); y = np.array([r["ddg_exp"] for r in tr])
            mu, sd = X.mean(0), X.std(0) + 1e-9
            A = np.column_stack([np.ones(len(X)), (X - mu) / sd]); R = 1.0 * np.eye(A.shape[1]); R[0, 0] = 0
            w = np.linalg.solve(A.T @ A + R, A.T @ y)
            Xe = np.array([[r[f] for f in te[0].keys() if f in fs] for r in te]) if False else \
                np.array([[r[f] for f in fs] for r in te])
            pred = np.column_stack([np.ones(len(Xe)), (Xe - mu) / sd]) @ w
            percx_h.append(spearmanr(pred, [r["ddg_exp"] for r in te]).statistic)
        print(f"  {nm:<26} per-cx mean Spearman = {np.mean(percx_h):+.3f}  (vs ref2015 {mref:+.3f})")
    print(f"\n  >> hybrid mean > ref2015 mean ({mref:+.3f}) leave-complex-out = BEAT FlexPepDock")


if __name__ == "__main__":
    main()
