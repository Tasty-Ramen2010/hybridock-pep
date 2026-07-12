"""E28 eval — our ensemble vs PPI-Affinity et al. on the 98-complex balanced benchmark.

Tests, all kcal/mol:
  diversity   : SS-class + length distribution (is it balanced, unlike crystal-65?)
  in-dist LOO : geometry+MJ LOO within the 98 (fair comparison vs PPI-Affinity 0.554/1.48)
  transfer    : fit crystal-65 -> predict the 98 (does our model GENERALIZE to a balanced set?)
  by SS class : LOO r within HELIX / SHEET / LOOP (where do we work?)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hybridock_pep.scoring.ensemble import GEOMETRY_FEATURES  # noqa: E402

feats = json.loads(Path("/tmp/e28_feats.json").read_text())
rows = list(feats.values())
y = np.array([r["y"] for r in rows])
X = np.array([[r.get(f, 0.0) for f in GEOMETRY_FEATURES] for r in rows])
L = np.array([r["L"] for r in rows])


def loo(X, y):
    p = np.zeros(len(y))
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd])
        w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        p[i] = np.r_[1, (X[i] - mu) / sd] @ w
    return p


def rr(p, yy):
    return pearsonr(p, yy).statistic, float(np.sqrt(((p - yy) ** 2).mean()))


print(f"=== 98-complex balanced benchmark (kcal/mol) ===")
print(f"  ΔG range [{y.min():.1f},{y.max():.1f}] median {np.median(y):.1f} std {y.std():.2f}")
print(f"  SS: {dict(Counter(r['ss'] for r in rows))}")
print(f"  length: median {np.median(L):.0f} range [{L.min()},{L.max()}], >15mer={np.mean(L>15):.0%}")
print(f"  vs crystal-65: ~all helix/loop, ~0 sheets -> THIS is balanced\n")

print("=== in-distribution LOO (fair vs PPI-Affinity 0.554 / RMSE 1.48) ===")
p = loo(X, y); r, rmse = rr(p, y)
print(f"  ours geometry+MJ LOO: r={r:+.3f}  RMSE={rmse:.2f}  MAE={np.mean(np.abs(p-y)):.2f}")
print(f"  PPI-Affinity (same set): r=0.554  RMSE=1.48   | PRODIGY 0.13 | RF-Score 0.23")
print(f"  guess-mean RMSE={y.std():.2f}")

print("\n=== by SS class (where do we work?) ===")
for ss in ["HELIX", "SHEET", "LOOP"]:
    idx = [i for i, r_ in enumerate(rows) if r_["ss"] == ss]
    if len(idx) >= 5:
        pi = loo(X[idx], y[idx]); ri, ei = rr(pi, y[idx])
        print(f"  {ss:<6} n={len(idx):>2}  LOO r={ri:+.3f}  RMSE={ei:.2f}")

print("\n=== GENERALIZATION: fit crystal-65 -> predict the 98 ===")
geo = {r["pdb"].upper(): r for r in json.loads(Path("/tmp/e19_cr.json").read_text())}
mj = json.loads(Path("/tmp/e24_contact.json").read_text())
tr = [dict(geo[p], mj_contact=mj[p]["mj_contact"]) for p in geo if p in mj]
ytr = np.array([r["y"] for r in tr])
Xtr = np.array([[r.get(f, 0.0) for f in GEOMETRY_FEATURES] for r in tr])
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
A = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
w, *_ = np.linalg.lstsq(A, ytr, rcond=None)
pred = np.column_stack([np.ones(len(X)), (X - mu) / sd]) @ w
r, rmse = rr(pred, y)
print(f"  crystal-65 -> 98-set: r={r:+.3f}  RMSE={rmse:.2f}")
print(f"  (positive => generalizes; ~0/neg => crystal-65 calibration doesn't transfer)")
