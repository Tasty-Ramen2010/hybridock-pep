"""E330 — our scorer on the PDBbind ~900 peptide-Kd set, for the ref2015 head-to-head.

Two honest numbers:
  (A) Within-PDBbind 5-fold CV using our 16 structural features (GBT) — leakage-free ceiling
      of our physics on THIS distribution. This is the number directly comparable to ref2015
      (which is a training-free physics score, so no CV needed for it).
  (B) Length-stratified 5-fold as a robustness check.
Reports Pearson r, Spearman, RMSE, MAE (kcal/mol). Also prints the ref2015 numbers if the
e329 cache exists, on the SAME complexes, so the comparison is strictly matched.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
JSONL = ROOT / "data/pdbbind_peptides.jsonl"
REF = ROOT / "data/e329_ref2015_pdbbind.json"
FEATS = ['poc_n', 'poc_f_hyd', 'poc_f_arom', 'poc_net', 'poc_eis', 'bsa_hyd', 'sasa_hb',
         'sasa_sb', 'arom_cc', 'hb_count', 'mj_contact', 'strength_bur', 'rg_per_L',
         'org_density', 'cys_frac', 'mean_burial']


def metrics(y, p):
    r = pearsonr(p, y)[0]
    sp = spearmanr(p, y).statistic
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    mae = float(np.mean(np.abs(p - y)))
    return r, sp, rmse, mae


def main():
    rows = [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]
    X = np.array([[r[k] for k in FEATS] for r in rows], float)
    y = np.array([r["y"] for r in rows], float)
    pdbs = [r["pdb"] for r in rows]
    print(f"=== E330 our scorer on PDBbind peptide-Kd (n={len(rows)}) ===")
    print(f"  experimental ΔG: mean={y.mean():.2f}  std={y.std():.2f}  "
          f"(mean-predictor RMSE={y.std():.2f})\n")

    gbt = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                    subsample=0.8, random_state=0)
    kf = KFold(5, shuffle=True, random_state=0)
    pred = cross_val_predict(gbt, X, y, cv=kf)
    r, sp, rmse, mae = metrics(y, pred)
    print("(A) OUR 16 features, within-PDBbind 5-fold CV (GBT):")
    print(f"    Pearson r = {r:+.3f}   Spearman = {sp:+.3f}   RMSE = {rmse:.2f}   MAE = {mae:.2f}\n")

    # matched comparison against ref2015 on the exact same complexes
    if REF.exists():
        ref = {d["pdb"]: d for d in json.loads(REF.read_text())}
        idx = [i for i, p in enumerate(pdbs) if p in ref]
        if len(idx) >= 20:
            ys = y[idx]
            ours = pred[idx]
            rx = np.array([ref[pdbs[i]]["ros_ifdG"] for i in idx])
            print(f"--- MATCHED head-to-head on n={len(idx)} complexes scored by both ---")
            r, sp, rmse, mae = metrics(ys, ours)
            print(f"  OURS (16-feat CV):      r={r:+.3f}  rho={sp:+.3f}  RMSE={rmse:.2f}  MAE={mae:.2f}")
            # ref2015 is REU, not kcal/mol -> only correlation is meaningful
            rr = pearsonr(rx, ys)[0]; rsp = spearmanr(rx, ys).statistic
            print(f"  ref2015 ifdG (unrelaxed): r={rr:+.3f}  rho={rsp:+.3f}  "
                  f"(REU, no kcal/mol RMSE/MAE)")
        else:
            print(f"(ref2015 cache has only {len(idx)} matched — run e329 to completion)")
    else:
        print("(ref2015 cache data/e329_ref2015_pdbbind.json not present yet)")


if __name__ == "__main__":
    main()
