"""E302 — train IFP on EVERYTHING we have (consistent-pipeline pool) and test honestly.

"Everything with verified-consistent features" = 925 PDBbind crystals (data/e296_ifp_cache.json) + 48 T100
crystals (runs/t100_extract) = 973 unique complexes. Both geometry (compute_geometry_features) and IFP
(compute_ifp) are the SAME production code on both sources — verified to machine precision:
  * e296 cache IFP == production compute_ifp (max|Δ|=0.0 on spot-checks)
  * T100 stored geom == production compute_geometry_features (0/16 keys differ)
The 360 PPIKB-with-structures are a SUBSET of the 925, so already included (no double counting).

Test = pooled leave-RECEPTOR-out CV (GroupKFold by receptor sequence). The key question: when the T100 is
part of a big pooled IFP training set — but its own receptor is still held out — does it beat the cold
out-of-sample 0.225 (E300)? And does IFP beat geom-only at full scale?

Run: OMP_NUM_THREADS=1 ~/miniconda3/envs/score-env/bin/python experiments/e302_ifp_train_everything.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import os

import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from hybridock_pep.scoring.interaction_map import (
    IFP_FEATURE_ORDER,
    _CRYSTAL_GEOM_ORDER,
    compute_ifp,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEOM = list(_CRYSTAL_GEOM_ORDER)

_3TO1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
         "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
         "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def gbt() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=300, max_depth=3, learning_rate=0.05, l2_regularization=1.0, random_state=0
    )


def r_mae(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    m = ~(np.isnan(pred) | np.isnan(y))
    pred, y = pred[m], y[m]
    return (float(pearsonr(pred, y)[0]) if m.sum() >= 3 else float("nan"),
            float(np.mean(np.abs(pred - y))) if m.sum() else float("nan"))


def net_charge(seq: str) -> int:
    return sum((c in "KR") - (c in "DE") for c in seq.upper())


def rec_seq_from_pdb(path: str) -> str:
    out = []
    for ln in open(path):
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            out.append(_3TO1.get(ln[17:20].strip(), "X"))
    return "".join(out)


# ---- source 1: 925 PDBbind (e296 cache) -------------------------------------------------------------
cache = json.load(open(os.path.join(ROOT, "data/e296_ifp_cache.json")))
seen = set()
rows = []
for d in cache:
    pid = d["pdb"].lower()
    if pid in seen:
        continue
    seen.add(pid)
    rows.append({"pdb": pid, "geom": d["x"], "ifp": d["ifp"], "y": d["y"],
                 "q": d["q"], "rseq": d["rseq"], "src": "pdbbind"})

# ---- source 2: 48 T100 (runs/t100_extract), same production geom+ifp ---------------------------------
t100 = {json.loads(l)["pdb"].lower(): json.loads(l)
        for l in open(os.path.join(ROOT, "data/t100_extra_features.jsonl"))}
for pid, rec in t100.items():
    if pid in seen:
        continue
    pep = sorted(glob.glob(os.path.join(ROOT, f"runs/t100_extract/{pid}_*_pep.pdb")))
    rcp = sorted(glob.glob(os.path.join(ROOT, f"runs/t100_extract/{pid}_*_rec.pdb")))
    if not pep or not rcp:
        continue
    seq = rec["seq"]
    g = {**rec, "length": len(seq)}
    try:
        f = compute_ifp(rcp[0], pep[0])
    except Exception as e:  # noqa: BLE001
        print(f"  IFP failed {pid}: {e}")
        continue
    seen.add(pid)
    rows.append({"pdb": pid, "geom": [float(g[k]) for k in GEOM],
                 "ifp": [float(f[k]) for k in IFP_FEATURE_ORDER], "y": rec["y"],
                 "q": net_charge(seq), "rseq": rec_seq_from_pdb(rcp[0]), "src": "t100"})

print(f"pooled IFP-computable complexes: {len(rows)} "
      f"(pdbbind {sum(r['src']=='pdbbind' for r in rows)}, t100 {sum(r['src']=='t100' for r in rows)})",
      flush=True)

geom = np.array([r["geom"] for r in rows], float)
ifp = np.array([r["ifp"] for r in rows], float)
y = np.array([r["y"] for r in rows], float)
q = np.array([r["q"] for r in rows], float)
src = np.array([r["src"] for r in rows])
grp = np.array([int(hashlib.md5(r["rseq"].encode()).hexdigest()[:8], 16) for r in rows])
charged = np.abs(q) >= 2


def loro(M: np.ndarray) -> np.ndarray:
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(8).split(M, y, grp):
        p[te] = gbt().fit(M[tr], y[tr]).predict(M[te])
    return p


pred_geom = loro(geom)
pred_ifp = loro(np.hstack([geom, ifp]))

t100m = src == "t100"
pdbm = src == "pdbbind"


def line(tag: str, pred: np.ndarray, mask: np.ndarray) -> str:
    ra, ma = r_mae(pred[mask], y[mask])
    rc, _ = r_mae(pred[mask & charged], y[mask & charged])
    return f"{tag:<34} r={ra:>6.3f}  MAE={ma:>5.2f}  charged r={rc:>6.3f}  (n={int(mask.sum())})"


print(f"\nTRAIN ON EVERYTHING (n={len(rows)}), pooled leave-receptor-out CV\n")
print("POOLED (all 973):")
print(" ", line("geom only (17)", pred_geom, np.ones(len(y), bool)))
print(" ", line("geom + IFP (36)", pred_ifp, np.ones(len(y), bool)))
print("\nPDBbind rows only:")
print(" ", line("geom only", pred_geom, pdbm))
print(" ", line("geom + IFP", pred_ifp, pdbm))
print("\nT100 rows only — held out by receptor, but trained alongside everything:")
print(" ", line("geom only", pred_geom, t100m))
print(" ", line("geom + IFP", pred_ifp, t100m))
print("   reference: cold OOS (E300) geom+IFP r=0.225 ; PPI-Affinity (in-dist) r=0.549")

out = {"n": len(rows), "pooled": {"geom": r_mae(pred_geom, y), "ifp": r_mae(pred_ifp, y)},
       "t100": {"geom": r_mae(pred_geom[t100m], y[t100m]), "ifp": r_mae(pred_ifp[t100m], y[t100m])},
       "pdbbind": {"geom": r_mae(pred_geom[pdbm], y[pdbm]), "ifp": r_mae(pred_ifp[pdbm], y[pdbm])}}
json.dump(out, open(os.path.join(ROOT, "data/e302_ifp_everything.json"), "w"), indent=1)
print(f"\nwrote data/e302_ifp_everything.json  (n={len(rows)})")
