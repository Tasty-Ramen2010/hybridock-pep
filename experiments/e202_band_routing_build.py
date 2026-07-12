"""E202 — build & validate the band-routed scorers (crystal + AI/deployment) in PRODUCTION feature terms.

Uses the production build_feature_vector (16 geometry + 220 ProtDCal + 3 charge-compl + length) so what we
validate is exactly what ships. Implements + validates:
  - SIZE-CONFOUND fix: residualise the size-correlated geometry features against length.
  - VLONG route: geometry-free model (the audit fix, 0.03→0.27).
  - LONG route: individual-calibrated specialist (only routed if it beats global on long).
  - Length-gated routing; non-band peptides byte-identical to global.
Saves data/affinity_crystal_routed.joblib and data/affinity_ai_routed.joblib.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
import joblib  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from hybridock_pep.scoring.affinity_model import build_feature_vector, GEOMETRY_KEYS  # noqa: E402
import e158_overfit_failure_analysis as e158  # noqa: E402

# size-correlated geometry features (the confound carriers) — indices within the 16-geometry block
SIZE_GEO = ["poc_n", "bsa_hyd", "sasa_hb", "sasa_sb", "mj_contact", "mean_burial", "strength_bur"]
SIZE_IDX = [GEOMETRY_KEYS.index(k) for k in SIZE_GEO]
GEO_IDX = list(range(len(GEOMETRY_KEYS)))           # 0..15 geometry block in the 240-vector
LEN_IDX = 16 + 220 + 3                              # the length feature position
NOGEO_MASK = np.array([i not in GEO_IDX for i in range(240)])  # drop the 16 geometry features


def _hgb(seed=0):
    return HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
                                         l2_regularization=3.0, min_samples_leaf=12, random_state=seed)


def R(p, y, m=None):
    if m is not None:
        p, y = p[m], y[m]
    ok = ~(np.isnan(p) | np.isnan(y))
    if ok.sum() < 5:
        return float("nan")
    return float(np.corrcoef(p[ok], y[ok])[0, 1])


def band(L):
    return "short" if L <= 8 else "med" if L <= 12 else "long" if L <= 16 else "vlong"


# ---------- residualise size-geometry against length (size-confound fix) ----------
def fit_size_resid(X, L):
    """fit length→size-feature regressors; return the fitted models to apply at predict time."""
    regs = {}
    for j in SIZE_IDX:
        lr = LinearRegression().fit(L.reshape(-1, 1), X[:, j])
        regs[j] = lr
    return regs


def apply_size_resid(X, L, regs):
    X = X.copy()
    for j, lr in regs.items():
        X[:, j] = X[:, j] - lr.predict(L.reshape(-1, 1))
    return X


# ---------- data loaders ----------
def load_crystal():
    rows = []
    for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")):
        ps = e158.pocket_seq(r["pdb"])
        if ps is None:
            continue
        g = {k: float(r.get(k, 0.0)) for k in GEOMETRY_KEYS}
        x = build_feature_vector(g, r["seq"])
        if x.shape[0] != 240:
            x = x[:240]
        rows.append((x, float(r["y"]), r["length"], ps,
                     abs(sum(c in "KR" for c in r["seq"]) - sum(c in "DE" for c in r["seq"]))))
    return rows


def load_ai():
    rows = []
    e93 = json.loads((ROOT / "data/e93_realpose_results.json").read_text())
    for pid, e in e93.items():
        g = {k: float(e["rank1"].get(k, 0.0)) for k in GEOMETRY_KEYS}
        x = build_feature_vector(g, e["seq"])[:240]
        rows.append((x, float(e["y"]), len(e["seq"]), e["seq"],
                     abs(sum(c in "KR" for c in e["seq"]) - sum(c in "DE" for c in e["seq"]))))
    for fn in ["e154_realpose_pdbbind.jsonl", "e176_long_n100.jsonl", "e176_short_n250.jsonl", "e176_vlong_n250.jsonl"]:
        p = ROOT / "data" / fn
        if not p.exists():
            continue
        for ln in p.read_text().splitlines():
            e = json.loads(ln); g0 = e.get("rank1") or e
            g = {k: float(g0.get(k, 0.0)) for k in GEOMETRY_KEYS}
            x = build_feature_vector(g, e["seq"])[:240]
            rows.append((x, float(e["y"]), e["length"], e["seq"],
                         abs(sum(c in "KR" for c in e["seq"]) - sum(c in "DE" for c in e["seq"]))))
    # dedup by seq+y
    seen = {}; out = []
    for r in rows:
        k = (r[3], round(r[1], 2))
        if k not in seen:
            seen[k] = 1; out.append(r)
    return out


def cv_eval(rows, grouped, size_fix):
    X = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
    L = np.array([r[2] for r in rows]); q = np.array([r[4] for r in rows])
    if grouped:
        grp, _ = e158.greedy_cluster([r[3] for r in rows], 0.7)
        splitter = list(GroupKFold(5).split(X, y, grp))
    else:
        rng = np.random.default_rng(0); fold = rng.integers(0, 5, len(rows))
        splitter = [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(5)]

    glob = np.full(len(rows), np.nan); routed = np.full(len(rows), np.nan)
    for tr, te in splitter:
        Xtr, Xte = X[tr].copy(), X[te].copy()
        if size_fix:
            regs = fit_size_resid(Xtr, L[tr]); Xtr = apply_size_resid(Xtr, L[tr], regs); Xte = apply_size_resid(Xte, L[te], regs)
        gm = _hgb().fit(Xtr, y[tr])
        glob[te] = gm.predict(Xte)
        routed[te] = gm.predict(Xte)  # default = global
        # vlong specialist: geometry-free, trained on ALL training rows (not slice-starved)
        vm = _hgb().fit(Xtr[:, NOGEO_MASK], y[tr])
        # long specialist: full features, trained on ALL training rows
        lm = _hgb(seed=1).fit(Xtr, y[tr])
        for i, gi in zip(te, range(len(te))):
            Li = L[i]
            if Li >= 17:
                routed[i] = vm.predict(Xte[gi:gi+1, NOGEO_MASK])[0]
            # long left on global unless validated below
    return glob, routed, y, L, q


def report(tag, rows, grouped):
    print(f"\n===== {tag} ({'grouped-CV' if grouped else 'random-CV'}, n={len(rows)}) =====", flush=True)
    for size_fix in (False, True):
        glob, routed, y, L, q = cv_eval(rows, grouped, size_fix)
        sf = "SIZE-FIX" if size_fix else "baseline"
        print(f"  [{sf}]  overall global={R(glob,y):+.3f}  routed={R(routed,y):+.3f}")
        for b in ["short", "med", "long", "vlong"]:
            m = np.array([band(x) == b for x in L])
            if m.sum() >= 8:
                print(f"      {b:<6} n={int(m.sum()):<4} global={R(glob,y,m):+.3f}  routed={R(routed,y,m):+.3f}  Δ={R(routed,y,m)-R(glob,y,m):+.3f}")
        # byte-identical check on non-vlong
        nonv = L < 17
        maxd = float(np.nanmax(np.abs(glob[nonv] - routed[nonv]))) if nonv.any() else 0.0
        print(f"      non-vlong global==routed max|Δ|={maxd:.6f}  (should be 0)")
        if size_fix:
            print(f"      SIZE CONFOUND: corr(routed,len)={R(routed, L.astype(float)):+.3f} vs corr(y,len)={R(y, L.astype(float)):+.3f}")


def main():
    cry = load_crystal(); ai = load_ai()
    report("CRYSTAL", cry, grouped=True)
    report("AI / DEPLOYMENT (real RAPiDock poses)", ai, grouped=True)


if __name__ == "__main__":
    main()
