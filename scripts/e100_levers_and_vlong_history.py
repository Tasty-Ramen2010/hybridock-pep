"""E100 — answer Ram's question (was vlong ever ~0.59?) + try all 3 beat-PPI levers.

PART 1 — HISTORY: was vlong-band r ever high on the COMBINED crystal set?
  The documented 0.585/0.68 was POOLED across all lengths on crystal cr65+the98. Did vlong itself
  ever look good, or did pooled range-leverage hide a weak vlong band? Compute per-band crystal r on
  the combined 156 (and per-dataset). Decisive: is cr65-only vlong=0.10 an anomaly or the rule?

PART 2 — LEVERS on real cr65 poses (cache runs/e99_cache.json):
  L1  ML-best-5 LEAK-CLEAN  — retrain the pose ranker leave-one-complex-out (no test complex in its
       training) → re-grade. Confirms the 0.478 isn't inflated by the ranker having seen these poses.
  L2  MED-BAND DENOISE      — drop high-CV noise features (poc_net CV9.6, poc_eis 3.8) ± the crystal→real
       FLIPPERS (rg_per_L, org_density); per-band + pooled r.
  L3  SALT-BRIDGE / CHARGE  — vlong's only real signal is sasa_sb (r=-0.64). Test a charge/sb-aware
       variant and a vlong sign-fix; does it lift vlong without breaking the rest?
"""
from __future__ import annotations

import csv
import json
import os
import sys
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

PROD = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
        "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density", "cys_frac"]
CACHE = ROOT / "runs" / "e99_cache.json"


def band(L):
    L = int(L)
    return "short≤8" if L <= 8 else "med9-12" if L <= 12 else "long13-16" if L <= 16 else "vlong≥17"


def fit_ridge(rows, cols, lam=1.0):
    X = np.array([[r["feat"][c] for c in cols] for r in rows], float)
    y = np.array([r["y"] for r in rows])
    ok = ~np.isnan(X).any(1)
    X, y = X[ok], y[ok]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    R = np.eye(A.shape[1]) * lam
    R[0, 0] = 0
    return mu, sd, np.linalg.solve(A.T @ A + R, A.T @ y)


def predict(feat, cols, p):
    mu, sd, w = p
    x = np.array([feat[c] for c in cols], float)
    return float(np.r_[1.0, (x - mu) / sd] @ w)


def loo(rows, cols):
    pred = np.full(len(rows), np.nan)
    for i in range(len(rows)):
        tr = [rows[j] for j in range(len(rows)) if j != i]
        pred[i] = predict(rows[i]["feat"], cols, fit_ridge(tr, cols))
    return pred


def st(p, y):
    m = ~(np.isnan(p) | np.isnan(y))
    return (pearsonr(p[m], y[m])[0], float(np.sqrt(np.mean((p[m] - y[m]) ** 2))), int(m.sum())) if m.sum() > 4 else (np.nan, np.nan, int(m.sum()))


def per_band(rows, pred, y):
    Ls = np.array([r["length"] for r in rows])
    out = {}
    for b in ["med9-12", "long13-16", "vlong≥17"]:
        m = np.array([band(L) == b for L in Ls])
        out[b] = st(pred[m], y[m]) if m.sum() >= 4 else (np.nan, np.nan, int(m.sum()))
    return out


# ============ PART 1: HISTORY ============
def part1():
    rows = []
    for f in ["data/pooled_benchmark_train.csv", "data/pooled_benchmark_test.csv"]:
        for r in csv.DictReader(open(ROOT / f)):
            rows.append({"y": float(r["y"]), "length": int(float(r["length"])), "dataset": r["dataset"],
                         "feat": {c: float(r[c]) for c in PROD}})
    y = np.array([r["y"] for r in rows])
    print(f"=== PART 1 — COMBINED CRYSTAL set (n={len(rows)}): was vlong ever ~0.59? ===\n")
    pred = loo(rows, PROD)
    rall = st(pred, y)
    print(f"  POOLED all-lengths crystal LOO:  r={rall[0]:+.3f} RMSE={rall[1]:.2f}  (this is the documented ~0.585)")
    print("\n  PER-BAND crystal r on COMBINED 156:")
    pb = per_band(rows, pred, y)
    for b, (r, rm, n) in pb.items():
        print(f"     {b:<11} n={n:<3} r={r:+.3f} RMSE={rm:.2f}")
    print("\n  vlong PER-DATASET (does the98 widen the range and lift vlong vs cr65-only 0.10?):")
    for ds in ["cr65", "the98"]:
        sub = [r for r in rows if r["dataset"] == ds and band(r["length"]) == "vlong≥17"]
        if len(sub) >= 4:
            yy = np.array([r["y"] for r in sub])
            # within-this-subset LOO using full-set fit (deployment-like): fit on ALL others
            pr = np.array([predict(s["feat"], PROD, fit_ridge([r for r in rows if r is not s], PROD)) for s in sub])
            print(f"     {ds:<6} vlong n={len(sub):<3} r={st(pr,yy)[0]:+.3f}  y-range={yy.max()-yy.min():.1f} kcal")
    print("\n  → reading: if pooled=0.59 but per-band vlong is low on BOTH datasets, the '0.59 for vlong'")
    print("    was the POOLED number (cross-band range leverage), never a vlong-specific correlation.\n")


# ============ PART 2: LEVERS ============
def part2():
    data = json.loads(CACHE.read_text())
    print(f"=== PART 2 — LEVERS on real cr65 poses (cache n={len(data)}) ===\n")

    def mean_feat(poses):
        return {k: float(np.nanmean([p["feat"][k] for p in poses if p["feat"].get(k) is not None])) for k in PROD}

    # build aggregations
    def rows_for(sel):
        out = []
        for d in data:
            ps = sel(d["poses"])
            if ps:
                out.append({"pdb": d["pdb"], "y": d["y"], "length": d["length"], "feat": mean_feat(ps)})
        return out

    top5 = rows_for(lambda pl: pl[:5])
    mlbest5 = rows_for(lambda pl: sorted([x for x in pl if x["ml"] is not None], key=lambda x: x["ml"])[:5] or pl[:5])
    y5 = np.array([r["y"] for r in top5])
    ym = np.array([r["y"] for r in mlbest5])

    print("  BASELINES (LOO, full 16 feat):")
    for nm, rows, yy in [("top-5 (diffusion)", top5, y5), ("ML-best-5 (leaky)", mlbest5, ym)]:
        p = loo(rows, PROD)
        s = st(p, yy)
        pb = per_band(rows, p, yy)
        print(f"     {nm:<20} pooled r={s[0]:+.3f} | med {pb['med9-12'][0]:+.2f} long {pb['long13-16'][0]:+.2f} vlong {pb['vlong≥17'][0]:+.2f}")

    # ---- L1: ML-best-5 LEAK-CLEAN (LOCO ranker) ----
    print("\n  L1 — ML-best-5 LEAK-CLEAN (ranker retrained leave-one-complex-out):")
    # recompute raw ml features per pose + RMSD label for LOCO ranker
    from hybridock_pep.scoring import pose_ranker_ml as PRM
    import joblib
    b = joblib.load(PRM.DEFAULT_MODEL_PATH)
    phi, psi = b["phi_kde"], b["psi_kde"]
    # need raw ml feats + per-pose rmsd; cache has rmsd but not raw ml feats → recompute feats from pose files
    CAMP = ROOT / "runs" / "e93_realpose_campaign"
    Xall, yall, gid, refs = [], [], [], []
    for gi, d in enumerate(data):
        poses = sorted((CAMP / d["pdb"] / "poses").glob("pose_*.pdb"), key=lambda q: int(q.stem.split("_")[1]))[:25]
        for k, p in enumerate(poses):
            if k >= len(d["poses"]):
                break
            f = PRM.compute_features(p, phi, psi)
            rm = d["poses"][k]["rmsd"]
            if f and rm is not None and not np.isnan(rm):
                Xall.append(f); yall.append(rm); gid.append(gi); refs.append((gi, k))
    Xall = np.array(Xall); yall = np.array(yall); gid = np.array(gid)
    # LOCO: for each complex, train ranker on others, pick its best-5 poses
    clean_rows = []
    for gi, d in enumerate(data):
        tr = gid != gi
        if tr.sum() < 100:
            continue
        m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                          l2_regularization=1.0, min_samples_leaf=20, random_state=0).fit(Xall[tr], yall[tr])
        idx = [k for (g, k) in refs if g == gi]
        if not idx:
            continue
        scores = m.predict(Xall[gid == gi])
        order = np.argsort(scores)[:5]
        chosen = [d["poses"][idx[o]] for o in order if idx[o] < len(d["poses"])]
        if chosen:
            clean_rows.append({"pdb": d["pdb"], "y": d["y"], "length": d["length"], "feat": mean_feat(chosen)})
    yc = np.array([r["y"] for r in clean_rows])
    pc = loo(clean_rows, PROD)
    sc = st(pc, yc); pbc = per_band(clean_rows, pc, yc)
    print(f"     ML-best-5 LEAK-CLEAN  pooled r={sc[0]:+.3f} (n={sc[2]}) | med {pbc['med9-12'][0]:+.2f} long {pbc['long13-16'][0]:+.2f} vlong {pbc['vlong≥17'][0]:+.2f}")
    print(f"     (vs leaky 0.478 — gap = leak inflation)")

    # ---- L2: MED-BAND DENOISE (drop noisy / flipping features) ----
    print("\n  L2 — DROP noisy/flipping features (on ML-best-5 leak-clean rows):")
    variants = {
        "drop poc_net,poc_eis": [c for c in PROD if c not in ("poc_net", "poc_eis")],
        "drop +rg_per_L,org_density": [c for c in PROD if c not in ("poc_net", "poc_eis", "rg_per_L", "org_density")],
        "robust-only (low-CV 8)": ["poc_n", "poc_f_hyd", "sasa_hb", "sasa_sb", "mj_contact", "mean_burial", "strength_bur", "cys_frac"],
    }
    for nm, cols in variants.items():
        p = loo(clean_rows, cols); s = st(p, yc); pb = per_band(clean_rows, p, yc)
        print(f"     {nm:<26} pooled r={s[0]:+.3f} | med {pb['med9-12'][0]:+.2f} long {pb['long13-16'][0]:+.2f} vlong {pb['vlong≥17'][0]:+.2f}")

    # ---- L3: SALT-BRIDGE / CHARGE-AWARE vlong handling ----
    print("\n  L3 — vlong charge/salt-bridge handling (ML-best-5 leak-clean):")
    base_cols = [c for c in PROD if c not in ("poc_net", "poc_eis")]
    p_base = loo(clean_rows, base_cols)
    # variant: vlong gets a dedicated sasa_sb-led short model; others get base
    pred_route = p_base.copy()
    vl_idx = [i for i, r in enumerate(clean_rows) if r["length"] >= 17]
    sb_cols = ["sasa_sb", "bsa_hyd", "strength_bur"]
    for i in vl_idx:
        tr = [clean_rows[j] for j in range(len(clean_rows)) if j != i and clean_rows[j]["length"] >= 13]  # long+vlong train
        if len(tr) >= 6:
            pred_route[i] = predict(clean_rows[i]["feat"], sb_cols, fit_ridge(tr, sb_cols))
    s_route = st(pred_route, yc); pb_route = per_band(clean_rows, pred_route, yc)
    print(f"     base (drop noisy)          pooled r={st(p_base,yc)[0]:+.3f} | vlong {per_band(clean_rows,p_base,yc)['vlong≥17'][0]:+.2f}")
    print(f"     + vlong→saltbridge submodel pooled r={s_route[0]:+.3f} | med {pb_route['med9-12'][0]:+.2f} long {pb_route['long13-16'][0]:+.2f} vlong {pb_route['vlong≥17'][0]:+.2f}")
    print("\n  PPI-Affinity reference: r=0.554 (independent set). Crystal pooled ceiling ≈0.537-0.585.")


if __name__ == "__main__":
    part1()
    part2()
