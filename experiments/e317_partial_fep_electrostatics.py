"""E317 — Can a PARTIAL FEP (charged/desolvation only) work, and do cheap electrostatics recover the residual?

Ram's idea: don't compute full ΔG — let the fast scorer do the shape (it is accurate on neutral), and spend
expensive compute ONLY on the charged/desolvation term it misses. This tests the premise: does ANY single-
structure electrostatics descriptor (incl. new ones — charge-scaling derivative, linear-response ½ factor,
distance-dependent dielectric, frustration) correlate with the CHARGED RESIDUAL (y minus what the geometry
scorer predicts, leave-receptor-out)? If yes, a cheap bolt-on could work. If no, the signal is in the
reorganization/fluctuation → the partial FEP must sample (but only the cheap charging leg).

Result (n=40 charged complexes with local structures): all descriptors r≈0 vs the residual — no static
shortcut. BUT frustration predicts the residual MAGNITUDE (Spearman −0.55) → a triage flag for which complexes
need FEP. Run: OMP_NUM_THREADS=1 python experiments/e317_partial_fep_electrostatics.py
"""
from __future__ import annotations
import json, os, glob, hashlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache = json.load(open(os.path.join(ROOT, "data/e296_ifp_cache.json")))
protd = {json.loads(l)["pdb"].lower() for l in open(os.path.join(ROOT, "data/e180_protdcal3d.jsonl"))}
data = [d for d in cache if d["pdb"] in protd]
X = np.array([d["x"] for d in data]); y = np.array([d["y"] for d in data])
grp = np.array([int(hashlib.md5(d["rseq"].encode()).hexdigest()[:8], 16) for d in data])

resid = np.full(len(y), np.nan)
for tr, te in GroupKFold(8).split(X, y, grp):
    m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                      l2_regularization=1.0, random_state=0).fit(X[tr], y[tr])
    resid[te] = y[te] - m.predict(X[te])
residmap = {d["pdb"].lower(): resid[i] for i, d in enumerate(data)}
qmap = {d["pdb"].lower(): abs(float(d["q"])) for d in data}

byid: dict[str, list[str]] = {}
for f in glob.glob(os.path.join(ROOT, "datasets/**/*.pdb"), recursive=True):
    byid.setdefault(os.path.basename(f).lower()[:4], []).append(f)


def pick(pid, k):
    for f in byid.get(pid, []):
        if k in f:
            return f
    return None


CHG = {("LYS", "NZ"): 1., ("ARG", "NH1"): .5, ("ARG", "NH2"): .5,
       ("ASP", "OD1"): -.5, ("ASP", "OD2"): -.5, ("GLU", "OE1"): -.5, ("GLU", "OE2"): -.5}


def atoms(fn):
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


rows, fr, ares = [], [], []
for pid in [p for p in qmap if qmap[p] >= 2]:
    rf, pf = pick(pid, "_rec"), pick(pid, "_pep")
    if not rf or not pf or pid not in residmap or not np.isfinite(residmap[pid]):
        continue
    rec, pep = atoms(rf), atoms(pf)
    if not rec or not pep:
        continue
    scale = []
    for lam in (0.8, 1.0, 1.2):
        e = sum((lam * qp) * qr / (r * r) for qp, xp in pep for qr, xr in rec
                if (r := np.linalg.norm(xp - xr)) >= 1)
        scale.append(e)
    born = coul = 0.
    for qp, xp in pep:
        born += qp * qp * sum(1 for _, xr in rec if np.linalg.norm(xp - xr) < 8)
        for qr, xr in rec:
            r = np.linalg.norm(xp - xr)
            if r >= 1:
                coul += qp * qr / r
    frustration = abs(coul) * born
    rows.append((residmap[pid], scale[1], (scale[2] - scale[0]) / 0.4, born, coul, frustration, 0.5 * scale[1]))
    fr.append(frustration); ares.append(abs(residmap[pid]))
rows = np.array(rows); res = rows[:, 0]
fr = np.array(fr); ares = np.array(ares)

print(f"charged complexes with structures + residual: n={len(rows)} (residual std={res.std():.2f})")
print("Single-structure electrostatics vs the CHARGED RESIDUAL (can cheap physics recover what shape misses?):")
for j, name in [(1, "dist-dep-dielectric Coulomb (ε=r)"), (2, "charge-scaling dE/dλ (lin-resp slope)"),
                (3, "Born desolvation (q²·burial)"), (4, "vacuum Coulomb"),
                (5, "frustration |Coul|·desolv"), (6, "linear-response ½·V (Marcus)")]:
    print(f"  {name:40s}: r={pearsonr(rows[:, j], res)[0]:+.3f}")
print(f"\nTriage: frustration vs |residual| — Spearman={spearmanr(fr, ares).statistic:+.3f} "
      f"(low-fr |resid| {ares[np.argsort(fr)[:len(fr) // 3]].mean():.2f} vs high-fr "
      f"{ares[np.argsort(fr)[-len(fr) // 3:]].mean():.2f})")
print("VERDICT: no static shortcut for the value → partial FEP must sample (charging leg only); "
      "frustration flags WHICH complexes need it.")
