#!/usr/bin/env python
"""Train on cropped PPI interfaces, predict peptide affinity. The test that decides the idea.

THE CLAIM BEING TESTED. If a protein-protein interface and a protein-peptide interface are the
same kind of object once you crop away everything that is not the interface -- and the survey says
they are, at 18 vs 13 residues and 508 vs 568 contacts -- then SKEMPI's measured affinities are
training data for a PEPTIDE scorer. That would be the single largest change in this project's
data situation: 294 cropped complexes now, 5,595 mutant measurements behind them, against a
peptide corpus of ~1,500 that has capped every model we have fitted.

THE TEST SET IS AS CLEAN AS WE WILL EVER GET. The eighteen designed complexes, scored on the
paper's OWN AF2 structures -- no docking, no co-folding, no pose error of ours anywhere in the
chain -- against the K_d the paper measured. Nothing about these complexes appears in SKEMPI, and
they are de novo designs, so there is no homology route for leakage either. If a PPI-trained
model predicts their affinity, the transfer is real.

CONTROLS, because "trained on 294 points, tested on 18" is exactly where people fool themselves:
  mean baseline    predict the training mean for everything. Any honest model must beat it.
  shuffled labels  refit on permuted dG. Any structure this recovers is fitting noise.
  leave-one-out    on the PPI side itself, to show the model works at all before asking it to
                   travel.

Pearson and Spearman are both reported: r is what the calibration literature quotes, but with 18
points one outlier can carry it, and rank correlation is what a shortlist actually depends on.

Usage: ppi_to_peptide_transfer.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "scripts"))
PAPER = ROOT / "datasets/coventry/binders_paper"
PPI = ROOT / "logs/ppi_interface_features.jsonl"
PEP_OUT = ROOT / "logs/coventry_truecomplex_features.jsonl"
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
#: our binder file is not the paper's protein for these -- excluded from any labelled use
BAD_ID = {"pc11", "pc17", "pc18", "pc35"}
R_GAS = 0.0019872


def featurise_true_complexes() -> None:
    """Score the 18 designed complexes as given. No docking, no repack, no pose of ours."""
    if PEP_OUT.exists():
        return
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from coventry_refine import init_rosetta
    pr = init_rosetta()
    sf = pr.create_score_function("ref2015")
    import sel_features as SF
    SF._PR, SF._SF = pr, sf
    import ppi_interface_features as PIF
    PIF._PR, PIF._SF = pr, sf

    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    with PEP_OUT.open("w") as fh:
        for t in ORDER:
            rec, pep = PAPER / f"{t}_1b1.pdb", PAPER / f"{t}_peptide.pdb"
            if not (rec.exists() and pep.exists()):
                continue
            row = {"name": t}
            kd = truth.get((t, t), {}).get("kd_M")
            row["dG_kcal_mol"] = (R_GAS * 298 * math.log(float(kd))) if kd else None
            try:
                row["feat"] = PIF.score_raw(str(rec), str(pep))
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
            fh.write(json.dumps(row) + "\n")


def load(path: Path):
    rows = []
    for l in path.read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if "feat" in r and r.get("dG_kcal_mol") is not None:
                rows.append(r)
    return rows


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spear(a, b):
    return corr(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))


def main() -> None:
    featurise_true_complexes()
    from sklearn.linear_model import HuberRegressor, RidgeCV
    from sklearn.preprocessing import StandardScaler

    ppi = load(PPI)
    pep = [r for r in load(PEP_OUT) if r["name"] not in BAD_ID]
    if not ppi or not pep:
        print("missing features"); return

    # Scoring raw atoms means a handful of crystal interfaces carry real clashes a repack would
    # have relieved -- 14 of 294 exceed +50 REU, one at +2109. Left in, that single interface
    # dominates a least-squares fit and the MAE runs to 1e11. They are dropped, and the count is
    # reported, because "no-repack needs a clash filter" is itself part of the repack answer.
    CLASH = 50.0
    n0 = len(ppi)
    ppi = [r for r in ppi if r["feat"].get("i_total", 0.0) < CLASH]
    print(f"  dropped {n0 - len(ppi)}/{n0} PPI interfaces with i_total > +{CLASH:g} REU "
          f"(raw-crystal clashes; no repack was applied)")

    keys = sorted(set(ppi[0]["feat"]) & set(pep[0]["feat"]))
    # fa_dun, fa_intra_rep and p_aa_pp are ONE-BODY terms: in an interface decomposition
    # (complex - receptor - peptide) they cancel exactly and come out at ~1e-14. Standardising
    # them divides by that, turns float noise into z-scores of 10, and the fit chases it --
    # p_aa_pp was drawing a +0.41 weight out of pure rounding error. Drop anything constant.
    _pre = np.array([[r["feat"].get(k, 0.0) for k in keys] for r in ppi], float)
    keys = [k for k, s_ in zip(keys, np.nan_to_num(_pre).std(0)) if s_ > 1e-6]
    print(f"  kept {len(keys)} features after dropping identically-zero one-body terms")

    X = np.nan_to_num(np.array([[r["feat"].get(k, 0.0) for k in keys] for r in ppi]),
                      posinf=0, neginf=0)
    y = np.array([r["dG_kcal_mol"] for r in ppi], float)
    Xp = np.nan_to_num(np.array([[r["feat"].get(k, 0.0) for k in keys] for r in pep]),
                       posinf=0, neginf=0)
    yp = np.array([r["dG_kcal_mol"] for r in pep], float)

    print(f"train: {len(y)} cropped PPI interfaces, {len(keys)} shared features")
    print(f"       dG {y.min():.1f} to {y.max():.1f}, median {np.median(y):.2f} kcal/mol")
    print(f"test:  {len(yp)} designed peptide complexes on the paper's OWN structures")
    print(f"       dG {yp.min():.1f} to {yp.max():.1f}, median {np.median(yp):.2f} kcal/mol\n")

    print("STEP 1 — does the model work on PPI at all? (leave-one-out on the training set)\n")
    oof = np.zeros(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        sc = StandardScaler().fit(X[m])
        sc.scale_ = np.where(sc.scale_ < 1e-6, 1.0, sc.scale_)   # never divide by ~0
        mdl = HuberRegressor(alpha=1.0, max_iter=500).fit(sc.transform(X[m]), y[m])
        oof[i] = float(np.clip(mdl.predict(sc.transform(X[i:i + 1]))[0], -40, 10))
    print(f"  leave-one-out on PPI:  r = {corr(oof, y):+.3f}   rho = {spear(oof, y):+.3f}   "
          f"MAE = {np.abs(oof - y).mean():.2f} kcal/mol")
    print(f"  predicting the mean:   MAE = {np.abs(y - y.mean()).mean():.2f} kcal/mol")

    print("\nSTEP 2 — does it TRANSFER to peptides? (trained on PPI only, never on peptides)\n")
    sc = StandardScaler().fit(X)
    sc.scale_ = np.where(sc.scale_ < 1e-6, 1.0, sc.scale_)
    mdl = HuberRegressor(alpha=1.0, max_iter=500).fit(sc.transform(X), y)
    pred = mdl.predict(sc.transform(Xp))
    print(f"  {'model':<40}{'r':>8}{'rho':>8}{'MAE':>9}")
    print(f"  {'PPI-trained -> peptides':<40}{corr(pred, yp):>8.3f}{spear(pred, yp):>8.3f}"
          f"{np.abs(pred - yp).mean():>9.2f}")
    print(f"  {'predict the PEPTIDE mean':<40}{'-':>8}{'-':>8}"
          f"{np.abs(yp - yp.mean()).mean():>9.2f}")
    print(f"  {'raw ref2015 i_total alone':<40}"
          f"{corr([r['feat']['i_total'] for r in pep], yp):>8.3f}"
          f"{spear([r['feat']['i_total'] for r in pep], yp):>8.3f}{'-':>9}")

    rng = np.random.default_rng(0)
    sh = []
    for _ in range(200):
        yy = rng.permutation(y)
        m2 = HuberRegressor(alpha=1.0, max_iter=500).fit(sc.transform(X), yy)
        sh.append(corr(m2.predict(sc.transform(Xp)), yp))
    sh = np.array(sh)
    real = corr(pred, yp)
    print(f"\n  shuffled-label control: r = {sh.mean():+.3f} +/- {sh.std():.3f}, "
          f"|r| >= real in {int((np.abs(sh) >= abs(real)).sum())}/200 draws "
          f"(empirical p = {(np.abs(sh) >= abs(real)).mean():.3f})")

    rho = spear(pred, yp)
    mae_ok = np.abs(pred - yp).mean() < np.abs(yp - yp.mean()).mean()
    sig = (np.abs(sh) >= abs(real)).mean() < 0.05
    if abs(real) > 0.35 and sig and abs(rho) > 0.35 and mae_ok:
        v = "TRANSFER IS REAL AND USABLE."
    elif abs(real) > 0.35 and sig:
        v = ("PEARSON ONLY. r beats its shuffle control but rho = "
             f"{rho:+.3f} and the MAE does not beat predicting the mean, so the correlation is "
             "carried by a couple of points, not by a usable ranking. Not shippable at n=14.")
    else:
        v = "no transfer at this scale."
    print("\n  -> " + v)
    print("\n  top weights (standardised):")
    o = np.argsort(-np.abs(mdl.coef_))
    for k in o[:8]:
        print(f"    {keys[k]:<28}{mdl.coef_[k]:+.3f}")


if __name__ == "__main__":
    main()
