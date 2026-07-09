"""E365b — score the external Wang complexes the REGULAR way (leave-cluster-out CV) and dissect the failures.

(1) Regular method: one cross_val_predict(GroupKFold on 60%-id clusters) over the whole ppikb set; read off the
    out-of-fold predictions for the Wang complexes not in PDBbind-925. Same leakage-free method as everywhere else
    — no bespoke retrain. Confirms the number matches the hand-split holdout.
(2) Failure analysis: is the error random, or does it SCALE with something? Test signed error vs true ΔG (range
    compression / regression-to-mean), peptide length, net charge, and whether a single linear recalibration
    collapses the error.

Run: OMP_NUM_THREADS=1 LD_LIBRARY_PATH=$CONDA_PREFIX/lib python scripts/e365b_failure_analysis.py
"""
from __future__ import annotations
import json, os, sys, math
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e330_ours_pdbbind import cluster_by_identity, ID_THRESH  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R, T = 0.001987, 298.15


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
            out[r["PDB"]] = (float(r["pKd"]), str(r["Protein"]).strip(), str(r["Sequence"]).strip())
        except (ValueError, TypeError):
            pass
    return out


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
    nc = len(set(clu.tolist()))
    gbt = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0)
    oof = cross_val_predict(gbt, X, y, cv=GroupKFold(min(5, nc)), groups=clu)      # <- the REGULAR method
    oof_by_pdb = dict(zip(pdbs, oof))

    ext = [p for p in pdbs if p in wang and p not in pdbbind]
    exp = np.array([-R * T * math.log(10) * wang[p][0] for p in ext])
    pred = np.array([oof_by_pdb[p] for p in ext])
    err = pred - exp                       # signed error (+ = under-binding / not negative enough)
    lens = np.array([len(usable[p]["seq"]) for p in ext])
    chg = np.array([abs(usable[p].get("net_charge", 0) or 0) for p in ext])

    print(f"=== REGULAR leave-cluster-out CV, external Wang complexes (n={len(ext)}) ===")
    print(f"  MAE={np.abs(err).mean():.2f}  RMSE={np.sqrt((err**2).mean()):.2f}  "
          f"r={pearsonr(exp, pred)[0]:+.3f}  Spearman={spearmanr(exp, pred)[0]:+.3f}")

    print("\n=== Does the error SCALE with something? (signed error = pred - exp) ===")
    for name, v in [("true ΔG (exp)", exp), ("|net charge|", chg), ("peptide length", lens),
                    ("prediction", pred)]:
        r_, p_ = pearsonr(v, err)
        print(f"  corr(signed error, {name:16s}) = {r_:+.3f}   (p={p_:.1e})")

    # range-compression test: fit exp = a*pred + b  (slope < 1 => predictions too flat)
    a, b = np.polyfit(pred, exp, 1)
    recal = a * pred + b
    print(f"\n=== Range compression (regression to the mean) ===")
    print(f"  fit  exp = {a:.2f}*pred + {b:.2f}   (slope 1.0 = perfect scale; <1 = model compresses range)")
    print(f"  pred spread (std) = {pred.std():.2f}   vs   exp spread (std) = {exp.std():.2f}")
    print(f"  MAE before recal = {np.abs(err).mean():.2f}   ->   after single linear rescale = {np.abs(recal - exp).mean():.2f}")

    # who are the worst, and are they one family?
    df = pd.DataFrame({"pdb": ext, "protein": [wang[p][1] for p in ext], "peptide": [wang[p][2] for p in ext],
                       "exp": np.round(exp, 2), "pred": np.round(pred, 2), "signed_err": np.round(err, 2),
                       "abs_err": np.round(np.abs(err), 2), "len": lens, "abs_charge": chg}).sort_values("abs_err", ascending=False)
    print("\n=== 10 worst ===")
    print(df.head(10).to_string(index=False, max_colwidth=24))
    tox = df[df["protein"].str.contains("bungarotoxin", case=False)]
    print(f"\n  α-bungarotoxin family: n={len(tox)}  mean signed_err={tox['signed_err'].mean():+.2f}  "
          f"(all same sign => systematic, not noise)")
    print(f"  MAE excluding bungarotoxin: {df[~df.index.isin(tox.index)]['abs_err'].mean():.2f}")
    df.to_csv(ROOT / "data/hybridock_wang2024_external_regular.csv", index=False)
    print(f"\n  -> data/hybridock_wang2024_external_regular.csv")


if __name__ == "__main__":
    main()
