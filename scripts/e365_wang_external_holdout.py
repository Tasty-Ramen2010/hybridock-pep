"""E365 — score the Wang-2024 complexes that our scorer has NEVER seen (true external holdout).

These PDBs appear in Wang et al. 2024 (Curr. Med. Chem. 31:4127) tables SM_TableS1/S2 but are NOT in our
PDBbind-925 training set. Their structural features exist only in the ppikb 59-feature space (desc3d+pocket_pkf),
and they are themselves rows of ppikb_features.jsonl — so to make them genuinely unseen we TRAIN on ppikb with
these targets AND their entire 60%-identity clusters removed, then predict them. Ground-truth ΔG is Wang's
independently-published pK_d (ΔG = -RT ln10 · pKd), not any label our model trained on.

Run: OMP_NUM_THREADS=1 python scripts/e365_wang_external_holdout.py
"""
from __future__ import annotations
import json, os, sys, csv, math
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e330_ours_pdbbind import cluster_by_identity, ID_THRESH, metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R, T = 0.001987, 298.15


def load_wang():
    """PDB -> (pKd, protein, peptide, reference, in_S2_benchmark) from the two Wang tables."""
    def rd(f):
        df = pd.read_excel(ROOT / f, header=3)
        df.columns = [str(c).strip() for c in df.columns]
        df = df[df["PDB"].astype(str).str.match(r"^[0-9][A-Za-z0-9]{3}$", na=False)].copy()
        df["PDB"] = df["PDB"].str.upper()
        return df
    s1, s2 = rd("SM_TableS1.xls"), rd("SM_TableS2.xls")
    s2set = set(s2["PDB"])
    out = {}
    for _, r in pd.concat([s1, s2]).drop_duplicates("PDB").iterrows():
        try:
            pkd = float(r["pKd"])
        except (ValueError, TypeError):
            continue
        out[r["PDB"]] = dict(pkd=pkd, protein=str(r["Protein"]).strip(), peptide=str(r["Sequence"]).strip(),
                             reference=str(r["Reference"]).strip(), in_S2=r["PDB"] in s2set)
    return out


def main():
    wang = load_wang()
    pdbbind = {json.loads(l)["pdb"].upper() for l in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines() if l.strip()}
    rows = [json.loads(l) for l in (ROOT / "data/ppikb_features.jsonl").read_text().splitlines() if l.strip()]
    # keep usable rows (features + label + seq), dedup by PDB
    usable = {}
    for r in rows:
        if r.get("desc3d") and r.get("pocket_pkf") and r.get("y") is not None and r.get("seq"):
            usable[r["pdb"].upper()] = r

    # the external holdout: in Wang tables, NOT in pdbbind-925 training, features available here
    ext = [p for p in usable if p in wang and p not in pdbbind]
    print(f"External Wang complexes never in PDBbind-925 training, with features: n={len(ext)}")

    seqs = [usable[p]["seq"] for p in usable]
    pdbs = list(usable.keys())
    clu = cluster_by_identity(seqs, ID_THRESH)
    cl = dict(zip(pdbs, clu.tolist()))
    ext_clusters = {cl[p] for p in ext}                       # clusters we must exclude from training
    train = [p for p in usable if cl[p] not in ext_clusters]  # everything not sequence-similar to a holdout
    print(f"Training rows (ppikb minus holdout clusters): {len(train)}  | excluded clusters: {len(ext_clusters)}")

    X = lambda p: usable[p]["desc3d"] + usable[p]["pocket_pkf"]
    Xtr = np.nan_to_num(np.array([X(p) for p in train], float))
    ytr = np.array([float(usable[p]["y"]) for p in train])
    gbt = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0)
    gbt.fit(Xtr, ytr)

    Xte = np.nan_to_num(np.array([X(p) for p in ext], float))
    pred = gbt.predict(Xte)
    exp = np.array([-R * T * math.log(10) * wang[p]["pkd"] for p in ext])  # ground truth = Wang published pKd

    recs = []
    for i, p in enumerate(ext):
        w = wang[p]
        recs.append(dict(pdb=p, in_S2_benchmark=w["in_S2"], protein=w["protein"], peptide=w["peptide"],
                         pKd=round(w["pkd"], 3), exp_dG_kcal_mol=round(float(exp[i]), 2),
                         hybridock_pred_dG_kcal_mol=round(float(pred[i]), 2),
                         abs_error_kcal_mol=round(float(abs(pred[i] - exp[i])), 2), literature_reference=w["reference"]))
    df = pd.DataFrame(recs).sort_values("exp_dG_kcal_mol").reset_index(drop=True)
    r, sp, rmse, mae = metrics(exp, pred)
    print(f"\n=== TRUE EXTERNAL HOLDOUT (never trained on these, nor their seq-clusters) ===")
    print(f"  n={len(df)}  MAE={mae:.2f}  RMSE={rmse:.2f}  Pearson r={r:+.3f}  Spearman={sp:+.3f}")
    print(f"  within 1.0 kcal: {(df['abs_error_kcal_mol'] <= 1.0).mean()*100:.0f}%  |  within 2.0: {(df['abs_error_kcal_mol'] <= 2.0).mean()*100:.0f}%\n")
    print(df[["pdb", "in_S2_benchmark", "peptide", "exp_dG_kcal_mol", "hybridock_pred_dG_kcal_mol",
              "abs_error_kcal_mol", "protein"]].to_string(index=False, max_colwidth=26))
    df.to_csv(ROOT / "data/hybridock_wang2024_external60.csv", index=False)
    print(f"\n  -> data/hybridock_wang2024_external60.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
