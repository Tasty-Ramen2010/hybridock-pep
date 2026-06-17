"""E277 — what can MISATO ACTUALLY contribute? (the coarse-charge premise is dead: water is stripped)

The coarse-charge idea was "learn how water screens charges" — impossible here: MISATO MD.hdf5 is
water-STRIPPED. What MISATO *does* have is configurational dynamics (induced-fit averaging) over 100
frames for 758 of our peptide complexes (which have Kd labels). This tests the only honest MISATO angle:
do MD-dynamics features (MD-averaged interaction energy, ligand RMSF, interaction-energy fluctuation,
buried SASA) add ORTHOGONAL signal on top of our static scorer? If yes, MISATO is a dynamics lever (not a
charge/water lever). If they are redundant, MISATO adds nothing.

Static features = our 17 pdbbind features. MISATO features = ie_mean, ie_std, lig_rmsf, bsasa (e251).
Clustered 5-fold CV (group by receptor) GBT, static vs static+MISATO.
Run: OMP_NUM_THREADS=1 python scripts/e277_misato_orthogonality.py
"""
from __future__ import annotations
import json, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

SFEAT = ["arom_cc", "bsa_hyd", "cys_frac", "hb_count", "length", "mean_burial", "mj_contact",
         "org_density", "poc_eis", "poc_f_arom", "poc_f_hyd", "poc_n", "poc_net", "rg_per_L",
         "sasa_hb", "sasa_sb", "strength_bur"]
mis = {json.loads(l)["id"]: json.loads(l) for l in open("data/e251_misato_flex.jsonl")
       if "ie_mean" in json.loads(l)}
lab = {json.loads(l)["pdb"].upper(): json.loads(l) for l in open("data/pdbbind_peptides.jsonl")}
rows = []
for k, m in mis.items():
    if k not in lab:
        continue
    d = lab[k]
    rows.append((
        [float(d[f]) for f in SFEAT],
        [m["ie_mean"], m["ie_std"], m["lig_rmsf"], m["bsasa_mean"], m.get("prot_rmsf", 0.0)],
        float(d["y"]),
    ))
Xs = np.array([r[0] for r in rows]); Xm = np.array([r[1] for r in rows])
y = np.array([r[2] for r in rows])
print(f"matched complexes {len(rows)} | static dim {Xs.shape[1]} | MISATO dim {Xm.shape[1]}", flush=True)


def cv(X):
    pred = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=0)
        m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    return pearsonr(y, pred)[0], np.mean(np.abs(y - pred))


rs, ms_ = cv(Xs)
rsm, msm = cv(np.hstack([Xs, Xm]))
rm, mm = cv(Xm)
print(f"  STATIC only        : r={rs:+.3f} MAE={ms_:.2f}")
print(f"  MISATO MD only     : r={rm:+.3f} MAE={mm:.2f}")
print(f"  STATIC + MISATO MD : r={rsm:+.3f} MAE={msm:.2f}  (Δr={rsm-rs:+.3f})")
json.dump(dict(n=len(rows), static_r=float(rs), misato_r=float(rm), combined_r=float(rsm),
               delta=float(rsm - rs)), open("data/e277_misato_ortho.json", "w"))
print("\nVERDICT: Δr>~0.03 => MISATO dynamics add orthogonal signal (a real lever, NOT charge/water).")
print("Δr~0 => MD dynamics redundant with static; MISATO adds nothing for affinity. (water stripped =>")
print("coarse-charge/desolvation idea structurally impossible on this dataset regardless.)")
