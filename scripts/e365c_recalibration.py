"""E365c — can a non-linear recalibration fix the range-compression on the external Wang complexes?

We test whether mapping pred -> exp with a polynomial (deg 1/3/5) or isotonic regression reduces MAE, AFTER
removing the 4 α-bungarotoxin outliers (n 47 -> 43). Every recalibration is fit WITHOUT leakage three ways:

  A) INDEPENDENT calibration: fit the map on in-distribution ppikb (their OOF pred vs label), apply to the 43.
     This is the realistic deployment path — calibrate on data you have, apply to new complexes.
  B) LEAVE-ONE-OUT on the 43: fit on 42, predict the held-out one. Honest but tiny calibration set.
  C) IN-SAMPLE (leaky): fit on 43, score the same 43. Reported ONLY to show how misleading it is.

Run: OMP_NUM_THREADS=1 LD_LIBRARY_PATH=$CONDA_PREFIX/lib python scripts/e365c_recalibration.py
"""
from __future__ import annotations
import json, os, sys, math
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e330_ours_pdbbind import cluster_by_identity, ID_THRESH  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R, T = 0.001987, 298.15
MAE = lambda a, b: float(np.abs(np.asarray(a) - np.asarray(b)).mean())


def load_wang():
    def rd(f):
        df = pd.read_excel(ROOT / f, header=3)
        df.columns = [str(c).strip() for c in df.columns]
        df = df[df["PDB"].astype(str).str.match(r"^[0-9][A-Za-z0-9]{3}$", na=False)].copy()
        df["PDB"] = df["PDB"].str.upper()
        return df
    out = {}
    for _, r in pd.concat([rd("SM_TableS1.xls"), rd("SM_TableS2.xls")]).drop_duplicates("PDB").iterrows():
        try:
            out[r["PDB"]] = (float(r["pKd"]), str(r["Protein"]).strip())
        except (ValueError, TypeError):
            pass
    return out


def fit_apply(kind, xtr, ytr, xte):
    """Fit a 1-D recalibration map on (xtr->ytr), apply to xte. Clip input to training range (guards Runge blow-up)."""
    xte = np.clip(xte, xtr.min(), xtr.max())
    if kind == "iso":
        m = IsotonicRegression(out_of_bounds="clip").fit(xtr, ytr)
        return m.predict(xte)
    deg = int(kind)
    c = np.polyfit(xtr, ytr, deg)
    return np.polyval(c, xte)


def main():
    wang = load_wang()
    pdbbind = {json.loads(l)["pdb"].upper() for l in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines() if l.strip()}
    rows = [json.loads(l) for l in (ROOT / "data/ppikb_features.jsonl").read_text().splitlines() if l.strip()]
    usable = {}
    for r in rows:
        if r.get("desc3d") and r.get("pocket_pkf") and r.get("y") is not None and r.get("seq"):
            usable[r["pdb"].upper()] = r
    pdbs = list(usable.keys())
    seqs = [usable[p]["seq"] for p in pdbs]
    X = np.nan_to_num(np.array([usable[p]["desc3d"] + usable[p]["pocket_pkf"] for p in pdbs], float))
    y = np.array([float(usable[p]["y"]) for p in pdbs])
    clu = cluster_by_identity(seqs, ID_THRESH)
    gbt = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0)
    oof = dict(zip(pdbs, cross_val_predict(gbt, X, y, cv=GroupKFold(min(5, len(set(clu)))), groups=clu)))

    # external set, minus the 4 bungarotoxin
    ext = [p for p in pdbs if p in wang and p not in pdbbind and "bungarotoxin" not in wang[p][1].lower()]
    pred = np.array([oof[p] for p in ext])
    exp = np.array([-R * T * math.log(10) * wang[p][0] for p in ext])
    print(f"External Wang, bungarotoxin removed: n={len(ext)}  (was 47)")
    print(f"  RAW (no recal):  MAE={MAE(pred, exp):.2f}  r={pearsonr(pred, exp)[0]:+.3f}  "
          f"pred-std={pred.std():.2f} vs exp-std={exp.std():.2f}\n")

    # independent calibration data = in-distribution ppikb (not external), OOF pred vs their label
    ext_set = set(ext)
    calib = [p for p in pdbs if p not in ext_set and p not in {q for q in pdbs if q in wang and q not in pdbbind}]
    cx = np.array([oof[p] for p in calib]); cy = y[[pdbs.index(p) for p in calib]]

    print(f"{'recal map':10s} {'A: indep-calib':>15s} {'B: LOO on 43':>14s} {'C: in-sample(leaky)':>20s}")
    for kind in ["1", "3", "5", "iso"]:
        # A: independent calibration set
        a = MAE(fit_apply(kind, cx, cy, pred), exp)
        # B: leave-one-out on the 43
        loo = np.array([fit_apply(kind, np.delete(pred, i), np.delete(exp, i), pred[i:i+1])[0] for i in range(len(pred))])
        b = MAE(loo, exp)
        # C: in-sample (leaky, do not trust)
        c = MAE(fit_apply(kind, pred, exp, pred), exp)
        name = {"1": "linear", "3": "cubic", "5": "quintic", "iso": "isotonic"}[kind]
        print(f"{name:10s} {a:>15.2f} {b:>14.2f} {c:>20.2f}")
    print(f"\n  raw baseline (no recal) MAE = {MAE(pred, exp):.2f}  <- beat this HONESTLY (cols A/B), not col C")


if __name__ == "__main__":
    main()
