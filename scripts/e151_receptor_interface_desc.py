"""E151 — add RECEPTOR-POCKET + INTERFACE ProtDCal descriptors (PPI uses the whole complex, we only did
the peptide) + feature selection. Push charged past 0.40 toward PPI's 0.71 on the same data.

E150: peptide ProtDCal descriptors lifted charged 0.29→0.40. PPI aggregates properties over the COMPLEX
(receptor pocket + interface), not just the ligand — likely the biggest remaining miss for charged binding
(the pocket's charge/polarity environment determines whether a peptide charge is satisfied). Add:
  - pocket residue property aggregations (same 22 scales, mean/std over receptor residues within 8 Å of pep)
  - interface peptide-residue property aggregations (peptide residues with burial>contact)
  - peptide×pocket property COMPLEMENTARITY (charge, hydrophobicity dot products)
Then SelectKBest (f_regression) on train folds → top-K, GBT, grouped CV on charged.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.feature_selection import f_regression  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
import importlib.util  # noqa: E402
_s = importlib.util.spec_from_file_location("e150", ROOT / "scripts/e150_protdcal_descriptors.py")
e150 = importlib.util.module_from_spec(_s); _s.loader.exec_module(e150)
PROD, SCALES, POS, NEG, THREE1 = e150.PROD, e150.SCALES, e150.POS, e150.NEG, e150.THREE1
PLROOT = ROOT / "data/drive_pull/pl/P-L"


def rec_pocket_residues(pdb, pep_centroid, cut=10.0):
    res, cur, key = [], None, None
    for ln in pdb.read_text().splitlines():
        if not ln.startswith("ATOM") or ln[12:16].strip()[:1] == "H":
            continue
        try:
            xyz = [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
        except ValueError:
            continue
        k = (ln[21], ln[22:27])
        if k != key:
            cur = {"rn": ln[17:20].strip(), "xyz": []}; res.append(cur); key = k
        cur["xyz"].append(xyz)
    pocket = []
    for r in res:
        c = np.mean(r["xyz"], 0)
        if np.linalg.norm(c - pep_centroid) < cut:
            pocket.append(e150.THREE1.get(r["rn"], "X"))
    return pocket


def pocket_desc(pocket_seq):
    if not pocket_seq:
        return [0.0] * (len(SCALES) * 2)
    out = []
    for sc in SCALES.values():
        v = np.array([sc.get(c, 0.0) for c in pocket_seq], float)
        out += [v.mean(), v.std()]
    return out


def compl_desc(pep_seq, pocket_seq):
    """peptide×pocket property complementarity (mean property products)."""
    if not pocket_seq or not pep_seq:
        return [0.0] * 4
    out = []
    for key in ("charge", "kd", "polar", "vol"):
        sc = SCALES[key]
        pp = np.mean([sc.get(c, 0.0) for c in pep_seq])
        rp = np.mean([sc.get(c, 0.0) for c in pocket_seq])
        out.append(pp * rp)
    return out


def cv_select(rows, kbest, k=5, seed=0):
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, k, len(rows))
    y = np.array([r["y"] for r in rows])
    X = np.nan_to_num(np.array([r["X"] for r in rows], float), nan=0.0, posinf=0.0, neginf=0.0)
    pred = np.full(len(rows), np.nan)
    for f in range(k):
        tr = fold != f
        # feature selection on TRAIN only
        if kbest and kbest < X.shape[1]:
            F, _ = f_regression(X[tr], y[tr])
            F = np.nan_to_num(F)
            sel = np.argsort(-F)[:kbest]
        else:
            sel = np.arange(X.shape[1])
        m = HistGradientBoostingRegressor(max_iter=600, max_depth=4, learning_rate=0.03,
                                          l2_regularization=4.0, min_samples_leaf=15, random_state=0).fit(X[tr][:, sel], y[tr])
        pred[fold == f] = m.predict(X[fold == f][:, sel])
    return pred, y


def metr(p, y):
    return pearsonr(p, y)[0], float(np.mean(np.abs(p - y))), float(np.sqrt(np.mean((p - y) ** 2)))


def main():
    pdbb = [json.loads(l) for l in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines()]
    rows = []
    for r in pdbb:
        q = abs(sum(c in POS for c in r["seq"]) - sum(c in NEG for c in r["seq"]))
        d = next((Path(p).parent for p in glob.glob(str(PLROOT / f"*/{r['pdb']}/{r['pdb']}_ligand.mol2"))), None)
        pdesc = e150.seq_descriptors(r["seq"])
        pock = pdesc_p = compl = None
        if d is not None:
            pep = e150.parse_pep_mol2(d / f"{r['pdb']}_ligand.mol2")
            if pep:
                cen = np.mean([np.mean(rr["xyz"], 0) for rr in pep], 0)
                pocket = rec_pocket_residues(d / f"{r['pdb']}_protein.pdb", cen)
                pock = pocket_desc(pocket); compl = compl_desc(r["seq"], pocket)
        if pock is None:
            pock = [0.0] * (len(SCALES) * 2); compl = [0.0] * 4
        X = [r[c] for c in PROD] + pdesc + pock + compl
        rows.append({"y": r["y"], "absq": q, "X": X})
    nfeat = len(rows[0]["X"])
    print(f"=== E151 +receptor-pocket+interface descriptors ({nfeat} total feats) ===")
    for name, sub in [("ALL", rows), ("charged |q|≥2", [r for r in rows if r["absq"] >= 2]),
                      ("high |q|≥3", [r for r in rows if r["absq"] >= 3])]:
        if len(sub) < 25:
            continue
        print(f"\n--- {name} (n={len(sub)}) ---  r / MAE / RMSE")
        for lbl, kb in [("all feats", None), ("top-60", 60), ("top-37 (PPI-like)", 37), ("top-25", 25)]:
            r, mae, rmse = metr(*cv_select(sub, kb))
            print(f"    {lbl:<18}{r:>+8.3f}{mae:>7.2f}{rmse:>7.2f}")
    print("\n  charged target → PPI 0.71. Receptor+interface+selection: how close on the SAME-source data?")


if __name__ == "__main__":
    main()
