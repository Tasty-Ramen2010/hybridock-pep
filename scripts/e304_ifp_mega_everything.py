"""E304 — train IFP on EVERYTHING incl. the newly-built PPIKB structures, and test honestly.

Pools three consistent-pipeline sources (same compute_geometry_features + compute_ifp throughout):
  * 925 PDBbind crystals          (data/e296_ifp_cache.json)
  * 48 T100 crystals              (runs/t100_extract, geom+ifp recomputed)
  * ~N new PPIKB crystals         (data/e303_ppikb_ifp_cache.json — split from data/rcsb_full/, asserted
                                    peptide-chain identity; NOT in PDBbind-925, so genuinely new)
Dedup by PDB id. Test = pooled leave-RECEPTOR-out CV (GroupKFold by receptor sequence). Reports geom-only
vs geom+IFP, pooled + per-source, overall + charged — i.e. does MORE data make IFP help more?

Run: OMP_NUM_THREADS=1 ~/miniconda3/envs/score-env/bin/python scripts/e304_ifp_mega_everything.py
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


def r_mae(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    m = ~(np.isnan(p) | np.isnan(y))
    p, y = p[m], y[m]
    return (float(pearsonr(p, y)[0]) if m.sum() >= 3 else float("nan"),
            float(np.mean(np.abs(p - y))) if m.sum() else float("nan"))


def net_charge(seq: str) -> int:
    return sum((c in "KR") - (c in "DE") for c in seq.upper())


def rec_seq(path: str) -> str:
    return "".join(_3TO1.get(ln[17:20].strip(), "X")
                   for ln in open(path) if ln.startswith("ATOM") and ln[12:16].strip() == "CA")


rows, seen = [], set()

# source 1 — PDBbind 925
for d in json.load(open(os.path.join(ROOT, "data/e296_ifp_cache.json"))):
    pid = d["pdb"].lower()
    if pid in seen:
        continue
    seen.add(pid)
    rows.append({"geom": d["x"], "ifp": d["ifp"], "y": d["y"], "q": d["q"],
                 "rseq": d["rseq"], "src": "pdbbind"})

# source 2 — T100 48
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
    except Exception:  # noqa: BLE001
        continue
    seen.add(pid)
    rows.append({"geom": [float(g[k]) for k in GEOM], "ifp": [float(f[k]) for k in IFP_FEATURE_ORDER],
                 "y": rec["y"], "q": net_charge(seq), "rseq": rec_seq(rcp[0]), "src": "t100"})

# source 3 — new PPIKB (E303)
e303 = os.path.join(ROOT, "data/e303_ppikb_ifp_cache.json")
if os.path.exists(e303):
    for d in json.load(open(e303)):
        pid = d["pdb"].lower()
        if pid in seen:
            continue
        seen.add(pid)
        rows.append({"geom": d["geom"], "ifp": d["ifp"], "y": d["y"], "q": d["q"],
                     "rseq": d["rseq"], "src": "ppikb"})

src = np.array([r["src"] for r in rows])
print(f"pooled IFP complexes: {len(rows)}  "
      f"(pdbbind {int((src=='pdbbind').sum())}, t100 {int((src=='t100').sum())}, "
      f"ppikb-new {int((src=='ppikb').sum())})", flush=True)

geom = np.array([r["geom"] for r in rows], float)
ifp = np.array([r["ifp"] for r in rows], float)
y = np.array([r["y"] for r in rows], float)
q = np.array([r["q"] for r in rows], float)
grp = np.array([int(hashlib.md5(r["rseq"].encode()).hexdigest()[:8], 16) for r in rows])
charged = np.abs(q) >= 2


def loro(M: np.ndarray) -> np.ndarray:
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(8).split(M, y, grp):
        p[te] = gbt().fit(M[tr], y[tr]).predict(M[te])
    return p


pg, pi = loro(geom), loro(np.hstack([geom, ifp]))


def line(tag: str, pred: np.ndarray, mask: np.ndarray) -> str:
    ra, ma = r_mae(pred[mask], y[mask])
    rc, _ = r_mae(pred[mask & charged], y[mask & charged])
    return f"{tag:<22} r={ra:>6.3f}  MAE={ma:>5.2f}  charged r={rc:>6.3f}  (n={int(mask.sum())})"


allm = np.ones(len(y), bool)
print(f"\nTRAIN ON EVERYTHING incl. new PPIKB (n={len(rows)}), pooled leave-receptor-out CV\n")
for name, m in [("POOLED (all)", allm), ("PDBbind only", src == "pdbbind"),
                ("T100 only", src == "t100"), ("PPIKB-new only", src == "ppikb")]:
    if m.sum() < 3:
        continue
    print(f"{name}:")
    print(" ", line("geom only (17)", pg, m))
    print(" ", line("geom + IFP (36)", pi, m))

out = {"n": len(rows), "by_src": {s: int((src == s).sum()) for s in ["pdbbind", "t100", "ppikb"]},
       "pooled": {"geom": r_mae(pg, y), "ifp": r_mae(pi, y)}}
json.dump(out, open(os.path.join(ROOT, "data/e304_ifp_mega.json"), "w"), indent=1)
print(f"\nwrote data/e304_ifp_mega.json")
