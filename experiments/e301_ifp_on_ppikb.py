"""E301 — IFP vs PPI-clone on the PPIKB complexes that carry crystal structures (leakage-safe).

PPIKB itself was ingested as sequence/pocket descriptors only — no receptor/peptide crystal splits, so the
interaction map (IFP) cannot be computed on the raw PPIKB n=305 "fresh" set. BUT 360 PPIKB complexes overlap
PDBbind (which we DO have crystal structures + IFP for, in data/e296_ifp_cache.json). Those 360 carry:
  * IFP (19) + geometry (17) from the crystal structure (e296 cache),
  * ProtDCal-3D desc3d (37) — PPI-Affinity's feature class (PPI-clone), from data/ppikb_features.jsonl,
  * a PPIKB experimental label.
So we can run the honest head-to-head "ours+IFP vs PPI-clone" on PPIKB-labelled data after all.

Two modes, both leakage-safe:
  MODE A (head-to-head): leave-RECEPTOR-out CV *within* the 360 — symmetric, self-contained (no PDBbind-925
    leak because training stays inside the 360 and the test receptor is always held out). Compares
    ours-geom / ours+IFP / PPI-clone(desc3d) on identical complexes, CV, and PPIKB labels.
  MODE B (deployment x-check, ours only): train geom+IFP on the 565 PDBbind complexes NOT in PPIKB, predict
    the 360 cold. Tests a PDBbind-trained IFP model deployed on PPIKB. PPI-clone can't run here (no desc3d
    for the 565).

Run: OMP_NUM_THREADS=1 ~/miniconda3/envs/score-env/bin/python experiments/e301_ifp_on_ppikb.py
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gbt() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=300, max_depth=3, learning_rate=0.05, l2_regularization=1.0, random_state=0
    )


def r_mae(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    m = ~(np.isnan(pred) | np.isnan(y))
    pred, y = pred[m], y[m]
    return float(pearsonr(pred, y)[0]), float(np.mean(np.abs(pred - y)))


def loro(M: np.ndarray, y: np.ndarray, grp: np.ndarray) -> np.ndarray:
    """Leave-receptor-out out-of-fold predictions."""
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(min(8, len(set(grp)))).split(M, y, grp):
        p[te] = gbt().fit(M[tr], y[tr]).predict(M[te])
    return p


# ---- load IFP cache (PDBbind crystals: geom x, ifp, PDBbind y, rseq) ---------------------------------
cache = json.load(open(os.path.join(ROOT, "data/e296_ifp_cache.json")))
cache_by_pdb: dict[str, dict] = {}
for d in cache:
    cache_by_pdb.setdefault(d["pdb"].lower(), d)

# ---- load PPIKB (desc3d = PPI-clone features, PPIKB label, net_charge) -------------------------------
ppikb: dict[str, dict] = {}
for line in open(os.path.join(ROOT, "data/ppikb_features.jsonl")):
    r = json.loads(line)
    p = (r.get("pdb") or "").lower()
    if p and isinstance(r.get("desc3d"), list) and len(r["desc3d"]) == 37 and np.isfinite(r.get("y", np.nan)):
        ppikb.setdefault(p, r)

inter = sorted(set(cache_by_pdb) & set(ppikb))
print(f"PPIKB complexes with crystal structure + IFP + desc3d: {len(inter)}", flush=True)

geom = np.array([cache_by_pdb[p]["x"] for p in inter], float)          # 17
ifp = np.array([cache_by_pdb[p]["ifp"] for p in inter], float)         # 19
desc = np.array([ppikb[p]["desc3d"] for p in inter], float)           # 37 (PPI-clone)
y = np.array([ppikb[p]["y"] for p in inter], float)                   # PPIKB label
q = np.array([ppikb[p].get("net_charge", 0) for p in inter], float)
grp = np.array([int(hashlib.md5(cache_by_pdb[p]["rseq"].encode()).hexdigest()[:8], 16) for p in inter])
charged = np.abs(q) >= 2

results: dict[str, dict] = {}


def report(tag: str, pred: np.ndarray) -> None:
    results[tag] = {
        "all": r_mae(pred, y),
        "charged": r_mae(pred[charged], y[charged]) if charged.sum() >= 3 else None,
        "neutral": r_mae(pred[~charged], y[~charged]) if (~charged).sum() >= 3 else None,
    }


# ===== MODE A — leave-receptor-out head-to-head within the 360 ========================================
report("OURS geom (17)", loro(geom, y, grp))
report("OURS geom+IFP (36)", loro(np.hstack([geom, ifp]), y, grp))
report("PPI-clone desc3d (37)", loro(desc, y, grp))

# ===== MODE B — deployment: train geom+IFP on PDBbind NOT in PPIKB, predict the 360 (ours only) ========
ppikb_pdbs = set(inter)
train_rows = [d for p, d in cache_by_pdb.items() if p not in ppikb_pdbs]
Xtr = np.array([d["x"] + d["ifp"] for d in train_rows], float)
ytr = np.array([d["y"] for d in train_rows], float)
Xte = np.hstack([geom, ifp])
report("OURS geom+IFP (deploy, train=PDBbind\\PPIKB)", gbt().fit(Xtr, ytr).predict(Xte))

# ---- print ------------------------------------------------------------------------------------------
print(f"\nIFP vs PPI-clone on PPIKB-with-structures  (n={len(inter)}, charged={int(charged.sum())})")
print("All vs PPIKB experimental labels. MODE A = leave-receptor-out CV within the 360 (head-to-head).\n")
print(f"{'method':<40} {'r_all':>7} {'MAE':>6}   {'r_charged':>9} {'r_neutral':>9}")
print("-" * 78)
for name in ["OURS geom+IFP (36)", "PPI-clone desc3d (37)", "OURS geom (17)",
             "OURS geom+IFP (deploy, train=PDBbind\\PPIKB)"]:
    m = results[name]
    rc = f"{m['charged'][0]:>9.3f}" if m["charged"] else f"{'—':>9}"
    rn = f"{m['neutral'][0]:>9.3f}" if m["neutral"] else f"{'—':>9}"
    print(f"{name:<40} {m['all'][0]:>7.3f} {m['all'][1]:>6.2f}   {rc} {rn}")

json.dump({"n": len(inter), "charged": int(charged.sum()), "results": results},
          open(os.path.join(ROOT, "data/e301_ifp_ppikb.json"), "w"), indent=1)
print(f"\nwrote data/e301_ifp_ppikb.json  (n={len(inter)})")
