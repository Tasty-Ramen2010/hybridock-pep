"""E325 — rerun the N2 test on the growing charged pose-cloud dataset (e323 PDBbind + e324 PPIKB).

N2 (E318) was real but underpowered (charged n=24, r=−0.37, perm-p=0.074). The e323/e324 campaigns generate
⟨V_elec⟩/Var over the RAPiDock cloud for every charged PDBbind + PPIKB complex. Run this anytime to see whether
the ensemble ⟨V_elec⟩ ~ charged-residual signal holds up as n grows (the go/no-go for the cheap-ensemble lever
vs the full FEP charging leg).

Run: OMP_NUM_THREADS=1 python scripts/e325_n2_at_scale.py
"""
from __future__ import annotations
import json, os, hashlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, GroupKFold
from scipy.stats import pearsonr, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FN = os.path.join(ROOT, "data/e323_charged_clouds.jsonl")
GEOM = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
        "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density", "cys_frac"]

def _load(fn):
    out = []
    if os.path.exists(fn):
        for l in open(fn):
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:  # tolerate a partial trailing line (campaign still writing)
                pass
    return out


rows = _load(FN)
rows = [r for r in rows if r.get("rank1") and np.isfinite(r.get("mean_ve", np.nan))]
if len(rows) < 15:
    print(f"only {len(rows)} clouds so far — let the e323/e324 campaigns accumulate (need ≥15). "
          f"Check: wc -l {FN}")
    raise SystemExit

y = np.array([r["y"] for r in rows])
# geometry features = mean over available top5 poses (ensemble geometry, e93 convention)
G = np.array([[np.mean([p[k] for p in r["top5"]]) for k in GEOM] for r in rows])
mean_ve = np.array([r["mean_ve"] for r in rows])
var_ve = np.array([r["var_ve"] for r in rows])
q = np.array([abs(r.get("q", 0)) for r in rows])
src = np.array([r.get("source", "pdbbind") for r in rows])
# group by receptor pocket signature (rough): use pdb id (each is a distinct receptor here)
grp = np.array([int(hashlib.md5(r["pdb"].encode()).hexdigest()[:8], 16) for r in rows])
n = len(rows)
print(f"charged clouds: n={n}  (PDBbind={np.sum(src=='pdbbind')}, PPIKB={np.sum(src=='ppikb')})")

# leave-one-out geometry residual
resid = np.full(n, np.nan)
splitter = GroupKFold(min(8, len(set(grp)))) if len(set(grp)) >= 8 else KFold(min(8, n), shuffle=True, random_state=0)
split = splitter.split(G, y, grp) if isinstance(splitter, GroupKFold) else splitter.split(G)
for tr, te in split:
    m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                      l2_regularization=1.0, random_state=0).fit(G[tr], y[tr])
    resid[te] = y[te] - m.predict(G[te])
print(f"geometry-only residual std = {np.nanstd(resid):.2f} kcal/mol")

r_mean = pearsonr(mean_ve, resid)[0]
r_var = pearsonr(var_ve, resid)[0]
print(f"\nensemble ⟨V_elec⟩ (LIE β·⟨V⟩) vs residual : r={r_mean:+.3f}")
print(f"ensemble Var(V_elec) (reorg)     vs residual : r={r_var:+.3f}")

# bootstrap + permutation on ⟨V_elec⟩
rng = np.random.default_rng(0)
boot = [pearsonr(mean_ve[i], resid[i])[0] for i in (rng.integers(0, n, n) for _ in range(3000))]
perm = [pearsonr(mean_ve, rng.permutation(resid))[0] for _ in range(3000)]
print(f"  ⟨V_elec⟩ boot95%=[{np.percentile(boot,2.5):+.3f},{np.percentile(boot,97.5):+.3f}]  "
      f"perm-p={np.mean(np.abs(perm)>=abs(r_mean)):.3f}")

# does adding ensemble electrostatics improve LOO prediction of y?
def loo_r(X):
    p = np.full(n, np.nan)
    sp = (GroupKFold(min(8, len(set(grp)))).split(X, y, grp) if len(set(grp)) >= 8
          else KFold(min(8, n), shuffle=True, random_state=0).split(X))
    for tr, te in sp:
        m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=0).fit(X[tr], y[tr])
        p[te] = m.predict(X[te])
    return pearsonr(p, y)[0]

r_geom, r_aug = loo_r(G), loo_r(np.column_stack([G, mean_ve, var_ve]))
print(f"\nLOO r  geometry={r_geom:+.3f}  +ensemble-electrostatics={r_aug:+.3f}  (Δ {r_aug-r_geom:+.3f})")

# CONTROL: neutral clouds (e326) — the ⟨V_elec⟩~residual signal should be SPECIFIC to charged complexes
NEU = os.path.join(ROOT, "data/e323_neutral_clouds.jsonl")
if os.path.exists(NEU):
    nrows = [json.loads(l) for l in open(NEU)]
    nrows = [r for r in nrows if r.get("rank1") and np.isfinite(r.get("mean_ve", np.nan))]
    if len(nrows) >= 15:
        ny = np.array([r["y"] for r in nrows])
        nG = np.array([[np.mean([p[k] for p in r["top5"]]) for k in GEOM] for r in nrows])
        nmv = np.array([r["mean_ve"] for r in nrows])
        ngrp = np.array([int(hashlib.md5(r["pdb"].encode()).hexdigest()[:8], 16) for r in nrows])
        nres = np.full(len(ny), np.nan)
        for tr, te in GroupKFold(min(8, len(set(ngrp)))).split(nG, ny, ngrp):
            m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                              l2_regularization=1.0, random_state=0).fit(nG[tr], ny[tr])
            nres[te] = ny[te] - m.predict(nG[te])
        print(f"\nCONTROL (neutral, n={len(nrows)}): ⟨V_elec⟩ vs residual r={pearsonr(nmv, nres)[0]:+.3f} "
              "(should be ≈0 if the signal is charge-specific)")

print("VERDICT: " + ("N2 HOLDS at scale — cheap ensemble ⟨V_elec⟩ is a real charged lever."
                     if (abs(r_mean) > 0.20 and np.mean(np.abs(perm) >= abs(r_mean)) < 0.05)
                     else f"still building / marginal at n={n} — keep accumulating."))
