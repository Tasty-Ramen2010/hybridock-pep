"""E321 (concept N5) — is the charge-frustration TRIAGE flag real enough to ship?

E317 found: frustration = |Coulomb|*Born-desolvation predicts the MAGNITUDE of the charged residual at
Spearman -0.55 (n=40) — low-frustration charged complexes carry ~3x the scorer error. Can't fix the value, but
could route only the high-error charged cases to the expensive FEP leg. Before shipping a triage flag we need:
bootstrap CI, permutation null, and a train/test threshold that generalizes (not the in-sample -0.55).

Run: OMP_NUM_THREADS=1 python experiments/e321_frustration_triage_n5.py
"""
from __future__ import annotations
import json, os, glob, hashlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr

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
pick = lambda pid, k: next((f for f in byid.get(pid, []) if k in f), None)
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


fr, ares = [], []
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
    fr.append(abs(coul) * born); ares.append(abs(residmap[pid]))
fr = np.array(fr); ares = np.array(ares); n = len(fr)
rho = spearmanr(fr, ares).statistic
rng = np.random.default_rng(0)
boot = [spearmanr(fr[i], ares[i]).statistic for i in (rng.integers(0, n, n) for _ in range(5000))]
perm = [spearmanr(rng.permutation(fr), ares).statistic for _ in range(5000)]
lo, hi = np.percentile(boot, [2.5, 97.5])
pval = np.mean(np.abs(perm) >= abs(rho))
print(f"n={n}  frustration vs |residual|  Spearman={rho:+.3f}  boot95%=[{lo:+.3f},{hi:+.3f}]  perm-p={pval:.3f}")

# does a frustration THRESHOLD generalize? split-half: fit median-threshold on train, test error-separation
order = rng.permutation(n)
tr, te = order[:n // 2], order[n // 2:]
thr = np.median(fr[tr])
lo_err = ares[te][fr[te] <= thr].mean()
hi_err = ares[te][fr[te] > thr].mean()
print(f"held-out split: below-median-frustration |resid|={lo_err:.2f} vs above={hi_err:.2f} "
      f"(ratio {lo_err/max(hi_err,1e-6):.2f}x)")
print("VERDICT: " + (
    f"triage signal SURVIVES (CI excludes 0, perm-p={pval:.3f}, held-out separation {lo_err/max(hi_err,1e-6):.1f}x) "
    "— usable as a which-complex-needs-FEP router." if hi < 0 and pval < 0.10 and lo_err > hi_err else
    f"triage signal is FRAGILE at n={n} (CI/perm/holdout not all clean) — do NOT ship as a hard gate; "
    "revisit when the charged structure set grows."))
