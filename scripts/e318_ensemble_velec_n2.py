"""E318 (concept N2 + LIE adaptation) — does the ELECTROSTATIC FLUCTUATION over a real generative
pose cloud carry the charged residual, with NO MD?

LIE (Aqvist 1994) computes the electrostatic contribution as beta*<V_elec> — an ENSEMBLE MEAN of the
peptide-environment interaction energy (beta=0.5 = the linear-response 1/2). FEP's reorganization energy is
1/2*Var(V_elec). E317 showed a SINGLE structure gives r~0 (no ensemble → no fluctuation). N2 asks: RAPiDock
already emits a 100-pose generative cloud per complex — can <V_elec> (LIE mean) and Var(V_elec) (reorganization)
over that cloud supply the signal a single crystal structure cannot?

Substrate: e93 real-pose campaign = 65 complexes, 100 real RAPiDock poses each + pocket receptor, with Kd labels.
No force field, no MD: formal charges on Lys/Arg/Asp/Glu (same convention as E317), V_elec = Sum q_i q_j / r_ij.
Residual = y - leave-one-out geometry-model prediction on the e93 pocket/interface features.

Run: OMP_NUM_THREADS=1 python scripts/e318_ensemble_velec_n2.py
"""
from __future__ import annotations
import json, os, glob
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E93 = json.load(open(os.path.join(ROOT, "data/e93_realpose_results.json")))
CAMP = os.path.join(ROOT, "runs/e93_realpose_campaign")

GEOM_FEATS = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb",
              "sasa_sb", "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact",
              "rg_per_L", "org_density", "cys_frac"]
CHG = {("LYS", "NZ"): 1., ("ARG", "NH1"): .5, ("ARG", "NH2"): .5,
       ("ASP", "OD1"): -.5, ("ASP", "OD2"): -.5, ("GLU", "OE1"): -.5, ("GLU", "OE2"): -.5}
PEP_CHG = {"K": 1, "R": 1, "D": -1, "E": -1}


def charges(fn):
    out = []
    for l in open(fn):
        if l.startswith(("ATOM", "HETATM")):
            qc = CHG.get((l[17:20].strip(), l[12:16].strip()))
            if qc:
                try:
                    out.append((qc, np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])))
                except ValueError:
                    pass
    return out


def velec(pep, rec, ddd=False):
    e = 0.0
    for qp, xp in pep:
        for qr, xr in rec:
            r = np.linalg.norm(xp - xr)
            if r >= 1.0:
                e += qp * qr / (r * r if ddd else r)
    return e


rows = []
for cid, d in E93.items():
    recf = os.path.join(CAMP, cid, "poses_raw/poses_raw/poses_raw_protein_raw.pdb")
    posef = sorted(glob.glob(os.path.join(CAMP, cid, "poses/pose_*.pdb")))
    if not os.path.exists(recf) or len(posef) < 50 or "top5" not in d:
        continue
    rec = charges(recf)
    if not rec:
        continue
    ve = [velec(charges(pf), rec) for pf in posef]
    ve = [v for v in ve if v == v]
    if len(ve) < 50:
        continue
    ve = np.array(ve)
    netq = sum(PEP_CHG.get(a, 0) for a in d["seq"])
    g = [float(d["top5"][f]) for f in GEOM_FEATS]
    rows.append(dict(cid=cid, y=float(d["y"]), g=g, netq=netq,
                     mean_ve=float(ve.mean()), var_ve=float(ve.var()), std_ve=float(ve.std())))

print(f"e93 real generative clouds usable: n={len(rows)}")
y = np.array([r["y"] for r in rows])
G = np.array([r["g"] for r in rows])
mean_ve = np.array([r["mean_ve"] for r in rows])
var_ve = np.array([r["var_ve"] for r in rows])
netq = np.array([abs(r["netq"]) for r in rows])

# leave-one-out geometry residual (KFold-8; receptors are ~distinct here)
resid = np.full(len(y), np.nan)
kf = KFold(8, shuffle=True, random_state=0)
for tr, te in kf.split(G):
    m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                      l2_regularization=1.0, random_state=0).fit(G[tr], y[tr])
    resid[te] = y[te] - m.predict(G[te])

print(f"geometry-only LOO residual std = {resid.std():.2f} kcal/mol")
print("\nEnsemble electrostatics (over the pose cloud) vs the CHARGED RESIDUAL:")
print(f"  LIE-style  <V_elec> (ensemble mean)  vs resid : r={pearsonr(mean_ve, resid)[0]:+.3f}")
print(f"  reorg-like Var(V_elec) (fluctuation) vs resid : r={pearsonr(var_ve, resid)[0]:+.3f}")

chg = netq >= 2
print(f"\nCHARGED subset (|net q|>=2, n={chg.sum()}):")
if chg.sum() >= 8:
    print(f"  <V_elec>       vs resid : r={pearsonr(mean_ve[chg], resid[chg])[0]:+.3f}")
    print(f"  Var(V_elec)    vs resid : r={pearsonr(var_ve[chg], resid[chg])[0]:+.3f}")
    print(f"  Var(V_elec)    vs |resid| (triage) : Spearman={spearmanr(var_ve[chg], np.abs(resid[chg])).statistic:+.3f}")

# does adding ensemble electrostatics to the geometry model improve LOO prediction?
def loo_r(X):
    p = np.full(len(y), np.nan)
    for tr, te in KFold(8, shuffle=True, random_state=0).split(X):
        m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=0).fit(X[tr], y[tr])
        p[te] = m.predict(X[te])
    return pearsonr(p, y)[0]

r_geom = loo_r(G)
r_aug = loo_r(np.column_stack([G, mean_ve, var_ve]))
print(f"\nLOO Pearson r  geometry-only={r_geom:+.3f}  +ensemble-electrostatics={r_aug:+.3f}  (delta {r_aug-r_geom:+.3f})")
print("VERDICT: real generative pose cloud " +
      ("SUPPLIES" if abs(pearsonr(mean_ve, resid)[0]) > 0.25 or (r_aug - r_geom) > 0.03 else "does NOT supply")
      + " charged-residual signal that a single crystal structure could not (E317).")
