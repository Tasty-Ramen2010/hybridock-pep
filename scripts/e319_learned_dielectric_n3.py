"""E319 (concept N3 + MM-PBSA adaptation) — learn the local dielectric / the Coulomb-desolvation cancellation.

MM-PBSA literature (Genheden/Ryde; Wang 2021 "screening electrostatic energy"): charged binding sites are
over-stabilised because a FIXED low dielectric does not screen enough; the fix is a VARIABLE/higher dielectric
for high-charge sites. The catastrophe is Coulomb - desolvation ~= small difference of two large terms.

N3: instead of assuming epsilon, feed BOTH large terms (vacuum Coulomb AND Born desolvation) to an ML model
and let it LEARN epsilon(environment) = the cancellation, from data. Compared against:
 (a) each term linearly (E312/E317 already showed net = noise),
 (b) a fixed-dielectric combination Coulomb/eps - desolv for a grid of eps.
Substrate: the n=40 charged crystal set (same as E317). Honest expectation: n=40 is small; if the learned
cancellation shows nothing here it means the signal needs real APBS energies at scale (>=100), a curation task.

Run: OMP_NUM_THREADS=1 python scripts/e319_learned_dielectric_n3.py
"""
from __future__ import annotations
import json, os, glob, hashlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr

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
    return next((f for f in byid.get(pid, []) if k in f), None)


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


coul_l, born_l, res_l, grp_l = [], [], [], []
for pid in [p for p in qmap if qmap[p] >= 2]:
    rf, pf = pick(pid, "_rec"), pick(pid, "_pep")
    if not rf or not pf or pid not in residmap or not np.isfinite(residmap[pid]):
        continue
    rec, pep = atoms(rf), atoms(pf)
    if not rec or not pep:
        continue
    born = coul = 0.
    for qp, xp in pep:
        born += qp * qp * sum(1 for _, xr in rec if np.linalg.norm(xp - xr) < 8)
        for qr, xr in rec:
            r = np.linalg.norm(xp - xr)
            if r >= 1:
                coul += qp * qr / r
    coul_l.append(coul); born_l.append(born); res_l.append(residmap[pid])
    grp_l.append(int(hashlib.md5(rf.encode()).hexdigest()[:8], 16))

coul = np.array(coul_l); born = np.array(born_l); res = np.array(res_l); g = np.array(grp_l)
print(f"charged crystal set: n={len(res)} (residual std={res.std():.2f})")

# (b) fixed-dielectric grid: does ANY single eps make Coulomb/eps - k*desolv track the residual?
print("\nFixed-dielectric sweep  (Coulomb/eps - Born)  vs residual:")
best = (0, 0)
for eps in (1, 2, 4, 8, 20, 40, 80):
    val = coul / eps - born
    r = pearsonr(val, res)[0]
    best = max(best, (abs(r), eps), key=lambda t: t[0])
    print(f"  eps={eps:3d}:  r={r:+.3f}")

# (a)/(c) learned cancellation: feed BOTH terms, let the model learn eps(environment)
Xc = np.column_stack([coul, born])
pred = np.full(len(res), np.nan)
gk = GroupKFold(min(6, len(set(g))))
for tr, te in gk.split(Xc, res, g):
    m = HistGradientBoostingRegressor(max_iter=200, max_depth=2, learning_rate=0.05,
                                      l2_regularization=2.0, random_state=0).fit(Xc[tr], res[tr])
    pred[te] = m.predict(Xc[te])
r_learn = pearsonr(pred, res)[0]
print(f"\nLearned cancellation (HGB on [Coulomb, Born], leave-receptor-out): r={r_learn:+.3f}")
print("VERDICT: " + ("learned dielectric RECOVERS residual — escalate to real APBS at scale."
                     if r_learn > 0.25 else
                     "no fixed OR learned single-structure dielectric recovers the residual at n=40 → "
                     "consistent with E317 (signal is the ensemble fluctuation, not a static dielectric); "
                     "real APBS would need n>=100 to overturn this, a curation task, not a quick win."))
