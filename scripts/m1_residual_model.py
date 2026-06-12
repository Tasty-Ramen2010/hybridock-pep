"""M1 — ML residual-correction model on the ensemble physics baseline.

baseline = linear map of intensive ensemble energy <E_int>/L -> ΔG (the keepable e49 result).
residual = ΔG_exp - baseline ;  ML (ridge) learns residual from charge + composition features.
ΔG_pred  = baseline + ML_residual.

Validation (the honest part):
  * preview: within-crystal-65 leave-COMPLEX-out (fast go/no-go; within-dataset = optimistic).
  * REAL:    leave-DATASET-out (train crystal-65 -> test the-98 and reverse) once e49b lands.
Charge-stratified (cf>=0.3) reported separately — that's the floor we attack. Spearman (robust)
+ Pearson. Ridge (heavy regularization) matched to n~65-163; net_charge feature first.

Run: `m1_residual_model.py preview`  (crystal-65 only) | `m1_residual_model.py cross` (needs e49b).
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

# Residual-model features (NOT the baseline energy itself): charge tier + composition tier.
RESID_FEATS = ["net_charge", "charged_frac", "arom_frac", "bulky_frac", "pro_frac",
               "L", "bsa_hyd", "mj_contact", "strength_bur", "e_int_std", "minus_tds"]


def seqfeat(seq):
    L = max(1, len(seq))
    return dict(net_charge=seq.count("K") + seq.count("R") - seq.count("D") - seq.count("E"),
                charged_frac=sum(c in "DEKR" for c in seq) / L,
                arom_frac=sum(c in "FWYH" for c in seq) / L,
                bulky_frac=sum(c in "FWYLIM" for c in seq) / L,
                pro_frac=seq.count("P") / L, L=len(seq))


def build_crystal():
    bench = {r["pdb"].upper(): r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    e49 = json.loads(Path("/tmp/e49_ens_mmgbsa.json").read_text())
    rows = []
    for k, m in bench.items():
        if k not in e49 or not m.get("peptide_seq"):
            continue
        pose = ROOT / f"logs/crystal65_n100/cr_{k}/poses/pose_0.pdb"
        rec = ROOT / m["pocket_pdb"]
        if not pose.exists():
            continue
        g = compute_geometry_features(pose, rec) or {}
        ei = e49[k]["e_int_mean"]
        if not np.isfinite(ei) or abs(ei) > 1e8:
            continue
        seq = m["peptide_seq"]
        rows.append(dict(seqfeat(seq), pdb=k, y=m["dg_exp"], e_int=ei, L_=len(seq),
                         e_int_perL=ei / max(1, len(seq)), e_int_std=e49[k]["e_int_std"],
                         minus_tds=e49[k]["minus_tds"], cf=e49[k]["cf"],
                         bsa_hyd=g.get("bsa_hyd", 0.0), mj_contact=g.get("mj_contact", 0.0),
                         strength_bur=g.get("strength_bur", 0.0)))
    return rows


def build_the98():
    e49b = json.loads(Path("/tmp/e49b_the98.json").read_text())
    e28 = json.loads(Path("/tmp/e28_feats.json").read_text())
    work = Path("/tmp/ppep_work")
    rows = []
    for k, v in e49b.items():
        ei = v["e_int_mean"]
        if not np.isfinite(ei) or abs(ei) > 1e8:
            continue
        g = compute_geometry_features(work / f"{k}_pep.pdb", work / f"{k}_rec.pdb") or {}
        seq = v["seq"]
        rows.append(dict(seqfeat(seq), pdb=k, y=v["y"], e_int=ei, L_=len(seq),
                         e_int_perL=ei / max(1, len(seq)), e_int_std=v["e_int_std"],
                         minus_tds=v["minus_tds"], cf=v["cf"],
                         bsa_hyd=g.get("bsa_hyd", 0.0), mj_contact=g.get("mj_contact", 0.0),
                         strength_bur=g.get("strength_bur", 0.0)))
    return rows


def _fit_baseline(tr):
    """y ~ a*winsor(e_int_perL) + b on train; returns (a,b,lo,hi).

    MD makes repulsive-non-binder outliers (5EI3 e_int_perL≈+37 vs −2..−4) that wreck a least-
    squares slope. Winsorize to train 5/95 percentiles so the baseline fit is robust (the metric
    lesson, applied to the FIT not just the eval)."""
    x = np.array([r["e_int_perL"] for r in tr]); y = np.array([r["y"] for r in tr])
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    a, b = np.polyfit(np.clip(x, lo, hi), y, 1)
    return a, b, lo, hi


def _ridge(Xtr, ytr, Xte, lam=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr = (Xtr - mu) / sd; Zte = (Xte - mu) / sd
    A = np.column_stack([np.ones(len(Ztr)), Ztr])
    n_f = A.shape[1]; R = lam * np.eye(n_f); R[0, 0] = 0.0
    w = np.linalg.solve(A.T @ A + R, A.T @ ytr)
    return np.column_stack([np.ones(len(Zte)), Zte]) @ w


def _report(tag, y, base, full, cf):
    hi = cf >= 0.30
    def s(v, m): return spearmanr(v[m], y[m]).statistic if m.sum() > 3 else float("nan")
    def p(v, m): return pearsonr(v[m], y[m]).statistic if m.sum() > 3 else float("nan")
    allm = np.ones(len(y), bool)
    print(f"  [{tag}]  baseline(<E_int>/L)  Spearman all={s(base,allm):+.3f} charged={s(base,hi):+.3f}"
          f"  | Pearson all={p(base,allm):+.3f}")
    print(f"  [{tag}]  + ML residual        Spearman all={s(full,allm):+.3f} charged={s(full,hi):+.3f}"
          f"  | Pearson all={p(full,allm):+.3f}   (n={len(y)},chg={hi.sum()})")


def preview():
    rows = build_crystal()
    print(f"=== M1 PREVIEW: within-crystal-65 leave-COMPLEX-out (n={len(rows)}) ===")
    y = np.array([r["y"] for r in rows]); cf = np.array([r["cf"] for r in rows])
    base = np.zeros(len(rows)); full = np.zeros(len(rows))
    X = np.array([[r[f] for f in RESID_FEATS] for r in rows])
    for i in range(len(rows)):
        tr = [j for j in range(len(rows)) if j != i]
        a, b, lo, hi = _fit_baseline([rows[j] for j in tr])
        bpred_tr = np.array([a * np.clip(rows[j]["e_int_perL"], lo, hi) + b for j in tr])
        resid_tr = y[tr] - bpred_tr
        base[i] = a * np.clip(rows[i]["e_int_perL"], lo, hi) + b
        full[i] = base[i] + _ridge(X[tr], resid_tr, X[i:i+1])[0]
    _report("LCO", y, base, full, cf)
    print("  >> go/no-go: does +ML beat baseline? (within-dataset is OPTIMISTIC — real test = cross)")


def cross():
    cr = build_crystal()
    if not Path("/tmp/e49b_the98.json").exists():
        print("e49b not ready"); return
    b98 = build_the98()
    print(f"=== M1 REAL: leave-DATASET-out (crystal-65 n={len(cr)} <-> the-98 n={len(b98)}) ===")
    for trn, ten, nm in [(cr, b98, "train crystal-65 -> test the-98"),
                         (b98, cr, "train the-98 -> test crystal-65")]:
        a, b, lo, hi = _fit_baseline(trn)
        ytr = np.array([r["y"] for r in trn])
        Xtr = np.array([[r[f] for f in RESID_FEATS] for r in trn])
        resid_tr = ytr - np.array([a * np.clip(r["e_int_perL"], lo, hi) + b for r in trn])
        yte = np.array([r["y"] for r in ten]); cfte = np.array([r["cf"] for r in ten])
        base = np.array([a * np.clip(r["e_int_perL"], lo, hi) + b for r in ten])
        Xte = np.array([[r[f] for f in RESID_FEATS] for r in ten])
        full = base + _ridge(Xtr, resid_tr, Xte)
        print(f"\n {nm}")
        _report("DSO", yte, base, full, cfte)
    print("\n  >> SHIP only if +ML beats baseline cross-dataset on the CHARGED column.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "preview"
    (cross if mode == "cross" else preview)()
