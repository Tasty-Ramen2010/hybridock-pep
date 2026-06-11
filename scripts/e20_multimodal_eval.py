"""E20 — test the multimodal/nonlinear levers borrowed from PPI-Affinity & multimodal-PPI.

Levers: (1) ESM-2 LM embeddings as features (multimodal-PPI/BindPred), (2) nonlinear model
(PPI-Affinity SVM -> we use GradientBoosting/RF). Honest tests on crystal-65:

  A) crystal-65 LOO: geometry-linear (our 0.576) vs +ESM(PCA) vs nonlinear vs fusion+nonlinear
  B) DATA-LEVER test (sequence-only, no structure confound): train ESM->ΔG on PEPBI (326),
     predict crystal-65. Does the LM lever transfer WHEN given more data? This isolates
     whether the gap to 0.62 is RECIPE (fixable by us) or DATA VOLUME (not).

ESM embeddings from /tmp/esm_affinity.json. CPU. sklearn.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
POCK = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis"]
IFACE = ["bsa_hyd", "sasa_hb", "sasa_sb", "arom_cc", "hb_count"]
GEO = POCK + IFACE


def load():
    cr = json.loads(Path("/tmp/e19_cr.json").read_text())
    pb = json.loads(Path("/tmp/e18_pb.json").read_text())
    esm = json.loads(Path("/tmp/esm_affinity.json").read_text())
    return cr, pb, esm


def geo_mat(recs):
    return np.array([[r.get(f, 0.0) for f in GEO] for r in recs], float)


def esm_mat(recs, esm):
    return np.array([esm.get(r["seq"], [0.0] * 1280) for r in recs], float)


def loo_linear(X, y):
    pred = np.zeros(len(y))
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd])
        w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pred[i] = np.r_[1, (X[i] - mu) / sd] @ w
    return pred


def loo_model(Xfun, y, make_model):
    """Xfun(train_idx)->(Xtr, Xte_row builder); refits PCA+model per fold (no leakage)."""
    pred = np.zeros(len(y))
    n = len(y)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        Xtr, Xte = Xfun(tr, [i])
        m = make_model()
        m.fit(Xtr, y[tr])
        pred[i] = m.predict(Xte)[0]
    return pred


def rr(pred, y):
    return pearsonr(pred, y).statistic, float(np.sqrt(np.mean((pred - y) ** 2)))


def main():
    cr, pb, esm = load()
    y = np.array([r["y"] for r in cr])
    Xg = geo_mat(cr)
    Xe_all = esm_mat(cr, esm)
    has_esm = np.array([r["seq"] in esm for r in cr])
    print(f"crystal-65: {len(cr)} complexes, {has_esm.sum()} with ESM embedding\n")

    print("=== A) crystal-65 LOO (r / RMSE) ===")
    # geometry linear (our baseline)
    p = loo_linear(Xg, y); print(f"  geometry linear (OURS)        {rr(p,y)[0]:+.3f}  RMSE {rr(p,y)[1]:.2f}")

    # ESM PCA-k linear, PCA refit per fold
    for k in (5, 10):
        def xf(tr, te, k=k):
            pca = PCA(n_components=min(k, len(tr) - 1)).fit(Xe_all[tr])
            return pca.transform(Xe_all[tr]), pca.transform(Xe_all[te])
        p = loo_model(xf, y, lambda: Ridge(alpha=1.0))
        print(f"  ESM-only PCA{k} ridge          {rr(p,y)[0]:+.3f}  RMSE {rr(p,y)[1]:.2f}")

    # fusion: geometry + ESM-PCA, linear
    def xf_fus(tr, te, k=8):
        pca = PCA(n_components=min(k, len(tr) - 1)).fit(Xe_all[tr])
        Xtr = np.column_stack([Xg[tr], pca.transform(Xe_all[tr])])
        Xte = np.column_stack([Xg[te], pca.transform(Xe_all[te])])
        return Xtr, Xte
    p = loo_model(xf_fus, y, lambda: Ridge(alpha=1.0))
    print(f"  FUSION geo+ESM-PCA8 ridge     {rr(p,y)[0]:+.3f}  RMSE {rr(p,y)[1]:.2f}")

    # nonlinear on geometry (PPI-Affinity lever)
    def xf_geo(tr, te):
        return Xg[tr], Xg[te]
    for nm, mk in [("GBM", lambda: GradientBoostingRegressor(n_estimators=100, max_depth=2)),
                   ("RF", lambda: RandomForestRegressor(n_estimators=200, max_depth=4))]:
        p = loo_model(xf_geo, y, mk)
        print(f"  geometry {nm:<3} (nonlinear)        {rr(p,y)[0]:+.3f}  RMSE {rr(p,y)[1]:.2f}")
    # fusion + nonlinear
    p = loo_model(xf_fus, y, lambda: GradientBoostingRegressor(n_estimators=100, max_depth=2))
    print(f"  FUSION geo+ESM GBM            {rr(p,y)[0]:+.3f}  RMSE {rr(p,y)[1]:.2f}")

    print(f"\n  guess-the-mean RMSE = {y.std():.2f}")

    print("\n=== B) DATA-LEVER: train ESM->ΔG on PEPBI(326 seqs), predict crystal-65 (seq-only) ===")
    # sequence-only: immune to PEPBI structure (Rosetta-model) confound
    pb_seqy = [(r["seq"], r["y"]) for r in pb if r["seq"] in esm]
    # dedupe by seq (mean y)
    from collections import defaultdict
    d = defaultdict(list)
    for s, yy in pb_seqy:
        d[s].append(yy)
    pb_seqs = [s for s in d if s not in set(r["seq"] for r in cr)]  # exclude crystal seqs
    Xpb = np.array([esm[s] for s in pb_seqs]); ypb = np.array([np.mean(d[s]) for s in pb_seqs])
    crmask = has_esm
    Xcr = Xe_all[crmask]; ycr = y[crmask]
    print(f"  train n={len(pb_seqs)} PEPBI seqs -> test n={len(ycr)} crystal seqs")
    for k in (10, 20):
        pca = PCA(n_components=min(k, len(pb_seqs) - 1)).fit(Xpb)
        m = Ridge(alpha=5.0).fit(pca.transform(Xpb), ypb)
        pr = m.predict(pca.transform(Xcr))
        print(f"  ESM PCA{k} ridge  PEPBI->crystal  r={pearsonr(pr,ycr).statistic:+.3f}  RMSE={np.sqrt(np.mean((pr-ycr)**2)):.2f}")
    print("  >> if positive, the LM lever transfers with data -> worth scaling data.")
    print("  >> if ~0/neg, the gap to 0.62 is DATA VOLUME (thousands of complexes), not recipe.")


if __name__ == "__main__":
    main()
