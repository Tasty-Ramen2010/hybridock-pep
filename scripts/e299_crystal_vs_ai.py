"""E299 — crystal vs AI(docked)-pose IFP calibration: does the interaction-map gain SURVIVE docked poses?

For the 65 crystal-65 complexes (have crystal complex + RAPiDock docked pose + dg_exp):
  crystal IFP = rich_ifp(receptor pocket, CRYSTAL peptide)
  docked  IFP = rich_ifp(receptor pocket, DOCKED rank1 peptide)
Train an IFP->dg model LOO on CRYSTAL IFP; evaluate on crystal IFP (the ceiling) AND on docked IFP (the
deployment number). Also: how faithfully does the docked map reproduce the crystal map?
Run: OMP_NUM_THREADS=1 python scripts/e299_crystal_vs_ai.py
"""
from __future__ import annotations
import json, os, importlib.util, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("e296", os.path.join(ROOT, "scripts/e296_ifp_rigorous.py"))
e296 = importlib.util.module_from_spec(spec)
# load only the helper funcs (avoid running its build): exec but it has no __main__ guard around build()
import types
src = open(os.path.join(ROOT, "scripts/e296_ifp_rigorous.py")).read()
src = src.split("def build()")[0]  # keep only constants + receptor_atoms_and_seq + peptide_atoms + rich_ifp
mod = types.ModuleType("e296p"); mod.__dict__["__file__"]=os.path.join(ROOT,"scripts/e296_ifp_rigorous.py"); exec(compile(src, "e296p", "exec"), mod.__dict__)
receptor_atoms_and_seq = mod.receptor_atoms_and_seq
rich_ifp = mod.rich_ifp
POS_RES, NEG_RES, AROM_RES, HYD_RES, POL_RES = mod.POS_RES, mod.NEG_RES, mod.AROM_RES, mod.HYD_RES, mod.POL_RES


def pdb_pep_atoms(path):
    """typed peptide atoms (cls, xyz) from a peptide PDB (by residue+atom name)."""
    out = []
    for ln in open(path):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        res = ln[17:20].strip(); atom = ln[12:16].strip(); el = atom[0]
        try:
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        except ValueError:
            continue
        cls = None
        if res in POS_RES and atom in ("NZ", "NH1", "NH2", "NE"):
            cls = "pos"
        elif res in NEG_RES and atom in ("OD1", "OD2", "OE1", "OE2"):
            cls = "neg"
        elif el == "N":
            cls = "don"
        elif el == "O":
            cls = "acc"
        elif el == "C" and res in AROM_RES:
            cls = "aro"
        elif el == "C":
            cls = "hyd"
        if cls:
            out.append((cls, xyz))
    return out


bench = {d["pdb"].lower(): d for d in json.load(open(os.path.join(ROOT, "data/benchmark_crystal.json")))}
cry, dok, ys = [], [], []
ok = 0
for pdb, d in bench.items():
    crys = os.path.join(ROOT, f"data/rcsb_full/{pdb}.pdb")
    pose = os.path.join(ROOT, f"runs/e93_realpose_campaign/{pdb.upper()}/poses/pose_1.pdb")
    pocket = d.get("pocket_pdb"); peppdb = d.get("peptide_pdb")
    if not (os.path.exists(crys) and os.path.exists(pose)):
        continue
    try:
        rec_atoms, _ = receptor_atoms_and_seq(crys)            # receptor (whole crystal, typed)
        crys_pep = pdb_pep_atoms(peppdb) if peppdb and os.path.exists(peppdb) else None
        dok_pep = pdb_pep_atoms(pose)
        if crys_pep is None or not dok_pep or not rec_atoms:
            continue
        ci = rich_ifp(rec_atoms, crys_pep); di = rich_ifp(rec_atoms, dok_pep)
    except Exception:
        continue
    cry.append(ci); dok.append(di); ys.append(float(d["dg_exp"])); ok += 1
cry = np.array(cry); dok = np.array(dok); y = np.array(ys)
print(f"crystal-65: usable {ok} (crystal IFP + docked IFP + dg_exp)", flush=True)

# how faithfully does docked IFP reproduce crystal IFP (per-feature)?
fcorr = [pearsonr(cry[:, j], dok[:, j])[0] for j in range(cry.shape[1]) if np.std(cry[:, j]) > 0 and np.std(dok[:, j]) > 0]
print(f"docked-vs-crystal IFP feature reproduction: mean r={np.mean(fcorr):.3f} (1=perfect docking)")


def loo(train_ifp, test_ifp):
    p = np.full(len(y), np.nan)
    for tr, te in LeaveOneOut().split(train_ifp):
        m = HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.05,
                                          l2_regularization=2.0, random_state=0).fit(train_ifp[tr], y[tr])
        p[te] = m.predict(test_ifp[te])
    return p


pc = loo(cry, cry)        # train crystal, test crystal (ceiling)
pd_ = loo(cry, dok)       # train crystal, test docked (deployment)
pdd = loo(dok, dok)       # train docked, test docked (self-consistent deployment)
def rm(p):
    return (pearsonr(y, p)[0], float(np.mean(np.abs(y - p))))
print("\n=== IFP-only model, crystal-65 LOO ===")
print(f"  CRYSTAL IFP (ceiling)        r={rm(pc)[0]:+.3f} MAE={rm(pc)[1]:.2f}")
print(f"  DOCKED IFP (crystal-trained) r={rm(pd_)[0]:+.3f} MAE={rm(pd_)[1]:.2f}")
print(f"  DOCKED IFP (docked-trained)  r={rm(pdd)[0]:+.3f} MAE={rm(pdd)[1]:.2f}  <- the honest deployment number")
json.dump(dict(n=ok, ifp_feat_reproduction=float(np.mean(fcorr)),
               crystal_r=float(rm(pc)[0]), docked_xtrain_r=float(rm(pd_)[0]), docked_selftrain_r=float(rm(pdd)[0])),
          open(os.path.join(ROOT, "data/e299_crystal_vs_ai.json"), "w"))
print("\nVERDICT: docked-trained r close to crystal r => IFP gain SURVIVES docking (deploy on AI poses).")
print("saved data/e299_crystal_vs_ai.json")
