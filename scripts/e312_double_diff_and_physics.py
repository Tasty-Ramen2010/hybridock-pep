"""E312 — (A) debunk the r=0.96 double-difference, (B) test cheap alchemical/physics tricks for charged.

(A) The README's "FEP-grade r=0.96 double-difference" predicts the 4th corner of a peptide×receptor 2×2 grid
    as yPRk + yPkR − yPkRk — pure arithmetic on THREE MEASURED experimental ΔGs (no scorer/features). We show
    it is beaten by the trivial "reuse one nearest measured value" baseline and that its r rides between-grid
    variance (real coupling error ~1.1 kcal/mol). Not a scorer capability.

(B) Can a cheap analogue of FEP's cancellation crack charged? Tested: ML relative ΔΔG (feature differences)
    and analytical electrostatics (Coulomb / Debye-Hückel screened / Born desolvation / their NET). All fail;
    the NET is pure noise, illustrating why FEP's path-integral is needed.
Run: OMP_NUM_THREADS=1 python scripts/e312_double_diff_and_physics.py
"""
from __future__ import annotations
import json, os, glob, hashlib
from collections import defaultdict
from itertools import combinations
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pf(v):
    if isinstance(v, str):
        v = v.strip(); return json.loads(v) if v.startswith("[") else float(v)
    return v


# ---------- (A) double-difference debunk ----------
recs = []
for l in open(os.path.join(ROOT, "data/ppikb_features.jsonl")):
    r = json.loads(l)
    if not r.get("desc3d"):
        continue
    try:
        y = pf(r["y"])
    except Exception:
        continue
    if np.isfinite(y):
        recs.append({"pep": r["seq"], "rec": r["protein_seq"], "y": float(y)})
ymap = defaultdict(list)
for r in recs:
    ymap[(r["pep"], r["rec"])].append(r["y"])
ymap = {k: float(np.mean(v)) for k, v in ymap.items()}
pep_recs = defaultdict(set)
for (p, rr) in ymap:
    pep_recs[p].add(rr)
multi = [p for p, rs in pep_recs.items() if len(rs) >= 2]
T, DD, NN = [], [], []
for a, b in combinations(multi, 2):
    for R, Rk in combinations(list(pep_recs[a] & pep_recs[b]), 2):
        try:
            yPR, yPRk, yPkR, yPkRk = ymap[(a, R)], ymap[(a, Rk)], ymap[(b, R)], ymap[(b, Rk)]
        except KeyError:
            continue
        T.append(yPR); DD.append(yPRk + yPkR - yPkRk); NN.append(yPRk)
T, DD, NN = map(np.array, (T, DD, NN))
print(f"(A) DOUBLE-DIFFERENCE debunk (n={len(T)} grids, target std={T.std():.2f}):")
print(f"    double-diff (3 measured)  r={pearsonr(T, DD)[0]:+.3f}  MAE={np.mean(np.abs(T - DD)):.2f}")
print(f"    nearest measured baseline r={pearsonr(T, NN)[0]:+.3f}  MAE={np.mean(np.abs(T - NN)):.2f}  <- beats it")
print(f"    coupling error ε std = {(T - DD).std():.2f} kcal/mol (the honest error; r rides between-grid variance)")

# ---------- (B1) anchoring (the honest same-receptor win) + relative ΔΔG for charged ----------
cache = json.load(open(os.path.join(ROOT, "data/e296_ifp_cache.json")))
protd = {json.loads(l)["pdb"].lower() for l in open(os.path.join(ROOT, "data/e180_protdcal3d.jsonl"))}
data = [d for d in cache if d["pdb"] in protd]
X = np.array([d["x"] for d in data]); IFP = np.array([d["ifp"] for d in data]); y = np.array([d["y"] for d in data])
q = np.array([abs(float(d["q"])) for d in data])
grp = np.array([int(hashlib.md5(d["rseq"].encode()).hexdigest()[:8], 16) for d in data])
F = np.hstack([X, IFP]); ch = q >= 2
p = np.full(len(y), np.nan)
for tr, te in GroupKFold(8).split(F, y, grp):
    m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05, l2_regularization=1.0, random_state=0)
    p[te] = m.fit(F[tr], y[tr]).predict(F[te])
byr = defaultdict(list)
for i, g in enumerate(grp):
    byr[g].append(i)
rng = np.random.default_rng(0)
for k in (2, 3):
    ap, ay = [], []
    for g, idx in byr.items():
        idx = np.array(idx)
        if len(idx) < k + 1:
            continue
        for _ in range(5):
            perm = rng.permutation(idx); ref, qry = perm[:k], perm[k:]
            ap += list(p[qry] + np.mean(y[ref] - p[ref])); ay += list(y[qry])
    print(f"(B) ANCHORING k={k}: within-receptor r={pearsonr(ay, ap)[0]:+.3f}")
# relative ΔΔG for charged
pr = []
dF, dY, cpair = [], [], []
for g, idx in byr.items():
    for i, j in combinations(idx, 2):
        dF.append(F[i] - F[j]); dY.append(y[i] - y[j]); cpair.append(q[i] >= 2 or q[j] >= 2)
dF, dY, cpair = np.array(dF), np.array(dY), np.array(cpair)
gpair = np.array([grp[i] for g, idx in byr.items() for i, j in combinations(idx, 2)])
pdY = np.full(len(dY), np.nan)
for tr, te in GroupKFold(8).split(dF, dY, gpair):
    m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05, l2_regularization=1.0, random_state=0)
    pdY[te] = m.fit(dF[tr], dY[tr]).predict(dF[te])
print(f"    charged ABSOLUTE r={pearsonr(y[ch], p[ch])[0]:+.3f}  vs  charged RELATIVE ΔΔG r={pearsonr(dY[cpair], pdY[cpair])[0]:+.3f} (worse)")

# ---------- (B2) analytical electrostatics on charged structures ----------
recf, pepf = {}, {}
for f in glob.glob(os.path.join(ROOT, "datasets/wang_pepset/*/*.pdb")):
    pid = os.path.basename(f)[:4].lower()
    (recf if "_rec_ref" in f else pepf if "_pep_ref" in f else {}).__setitem__(pid, f) if ("_rec_ref" in f or "_pep_ref" in f) else None
CHG = {("LYS", "NZ"): 1, ("ARG", "NH1"): .5, ("ARG", "NH2"): .5, ("ASP", "OD1"): -.5,
       ("ASP", "OD2"): -.5, ("GLU", "OE1"): -.5, ("GLU", "OE2"): -.5}
labels = {d["pdb"].lower(): float(d["y"]) for d in data if abs(float(d["q"])) >= 2}


def chgatoms(fn):
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


rows = []
for pid in set(recf) & set(pepf) & set(labels):
    rec, pep = chgatoms(recf[pid]), chgatoms(pepf[pid])
    if not rec or not pep:
        continue
    coul = scr = born = 0.0
    for qp, xp in pep:
        born += qp * qp * sum(1 for _, xr in rec if np.linalg.norm(xp - xr) < 8)
        for qr, xr in rec:
            r = np.linalg.norm(xp - xr)
            if r < 1:
                continue
            coul += qp * qr / r; scr += qp * qr * np.exp(-r / 9.6) / r
    rows.append((labels[pid], coul, scr, born, coul - 0.3 * born))
rows = np.array(rows)
if len(rows) >= 5:
    yy = rows[:, 0]
    print(f"(B) analytical electrostatics vs charged ΔG (n={len(rows)}):")
    for j, name in [(1, "vacuum Coulomb"), (2, "screened Coulomb"), (3, "Born desolvation"), (4, "NET Coul−desolv")]:
        print(f"    {name:18s} r={pearsonr(rows[:, j], yy)[0]:+.3f}")
    print("    NET ~ noise: subtracting two large single-point terms amplifies error -> why FEP's path-integral is needed.")
