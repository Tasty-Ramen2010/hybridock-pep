"""E98 — forensic autopsy: why does real-pose deployment (r≈0.37) fall below crystal (0.585) and
the documented 0.486? Identifies which features de-correlate, quantifies pose noise, breaks down by
length, and compares pose-aggregation strategies (a) ML-ranker best-5 and (b) mean-predicted-ΔG.

CONTROL FIRST (decisive): re-grade the CRYSTAL oracle poses with THIS exact pipeline. If crystal
LOO ≈ 0.585, the grader is sound and the real-pose drop is real. If crystal LOO is also low, the
grader differs from the documented pipeline → the 0.37 is partly a script artifact.

Sections:
  0. CONTROL — crystal-pose LOO with this pipeline (must reproduce ~0.585)
  1. AGGREGATIONS — rank-1 / top-5 mean-feat / mean-pred-ΔG / ML-best-1 / ML-best-5 (LOO r/RMSE)
  2. PER-FEATURE DE-CORRELATION — Pearson(feat,y): crystal vs real; Δ ranks what broke
  3. POSE NOISE — within-complex CV of each feature across real poses (regression-dilution source)
  4. VALUE SHIFT — crystal→real mean shift per feature (in crystal-σ units): systematic miscalibration
  5. LENGTH BREAKDOWN — r/RMSE by short/med/long/vlong on the best aggregation
  6. RESIDUAL AUTOPSY — what predicts our error: features + pose-RMSD vs |error|
"""
from __future__ import annotations

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

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from Bio.PDB import PDBParser  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from hybridock_pep.scoring.geometry_features import (  # noqa: E402
    GEOMETRY_FEATURE_KEYS, compute_geometry_features,
)
from hybridock_pep.scoring import pose_ranker_ml as PRM  # noqa: E402

CAMP = ROOT / "runs" / "e93_realpose_campaign"
PROD = list(GEOMETRY_FEATURE_KEYS)
SHORT = ["bsa_hyd", "mj_contact", "strength_bur"]
P = PDBParser(QUIET=True)
NPOSE = 25  # poses per complex used for noise / ML-ranker / aggregation


def ca(pdb):
    m = P.get_structure("x", str(pdb))[0]
    return np.array([a.coord for ch in m for r in ch if r.id[0] == " " for a in r if a.name == "CA"])


def rmsd(a, b):
    n = min(len(a), len(b))
    if n < 3:
        return np.nan
    a, b = a[:n] - a[:n].mean(0), b[:n] - b[:n].mean(0)
    H = a.T @ b
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.sqrt(((a @ R.T - b) ** 2).sum(1).mean()))


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


def predict(feat, cols, params):
    mu, sd, w = params
    x = np.array([feat[c] for c in cols], float)
    return float(np.r_[1.0, (x - mu) / sd] @ w)


def loo(rows, router=True):
    pred = np.full(len(rows), np.nan)
    for i in range(len(rows)):
        tr = [rows[j] for j in range(len(rows)) if j != i]
        if router and rows[i]["length"] <= 8:
            trb = [r for r in tr if r["length"] <= 8]
            cols = SHORT if len(trb) >= 6 else PROD
            base = trb if len(trb) >= 6 else tr
        else:
            cols, base = PROD, tr
        pred[i] = predict(rows[i]["feat"], cols, fit_ridge(base, cols))
    return pred


def stat(p, y):
    m = ~(np.isnan(p) | np.isnan(y))
    if m.sum() < 5:
        return (np.nan, np.nan, np.nan, int(m.sum()))
    return (pearsonr(p[m], y[m])[0], spearmanr(p[m], y[m]).statistic,
            float(np.sqrt(np.mean((p[m] - y[m]) ** 2))), int(m.sum()))


def mean_feat(feats):
    out = {}
    for k in PROD:
        v = [f[k] for f in feats if f and f.get(k) is not None and not np.isnan(f[k])]
        out[k] = float(np.mean(v)) if v else np.nan
    return out if any(not np.isnan(v) for v in out.values()) else None


def main():
    bench = {r["pdb"]: r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    bundle = joblib.load(PRM.DEFAULT_MODEL_PATH)
    phi_kde, psi_kde, ml = bundle["phi_kde"], bundle["psi_kde"], bundle["model"]
    complexes = sorted([d.name for d in CAMP.iterdir() if (d / "poses").exists()])
    print(f"=== E98 real-pose forensic autopsy ({len(complexes)} complexes, {NPOSE} poses each) ===\n")

    data = []  # per-complex bundle
    for cx in complexes:
        meta = bench.get(cx)
        if not meta or meta.get("dg_exp") is None:
            continue
        receptor = Path(meta["pocket_pdb"]).resolve()
        y = float(meta["dg_exp"])
        length = int(meta.get("peptide_len") or len(meta.get("peptide_seq", "")))
        xtal_ca = ca(Path(meta["peptide_pdb"]))
        crystal_feat = compute_geometry_features(Path(meta["peptide_pdb"]), receptor)
        poses = sorted((CAMP / cx / "poses").glob("pose_*.pdb"),
                       key=lambda q: int(q.stem.split("_")[1]))[:NPOSE]
        pf = []  # (rmsd, ml_score, feat)
        for p in poses:
            f = compute_geometry_features(p, receptor)
            if not f:
                continue
            mlf = PRM.compute_features(p, phi_kde, psi_kde)
            ms = float(ml.predict(np.array([mlf]))[0]) if mlf else np.nan
            pf.append((rmsd(ca(p), xtal_ca), ms, f))
        if len(pf) < 3 or crystal_feat is None:
            continue
        data.append({"pdb": cx, "y": y, "length": length, "crystal": crystal_feat, "poses": pf})
        print(f"  {cx}: len={length} y={y:+.2f} poses={len(pf)} bestRMSD={min(r[0] for r in pf):.1f}Å", flush=True)

    n = len(data)
    print(f"\nusable complexes: {n}\n")
    yv = np.array([d["y"] for d in data])

    # ---------- 0. CONTROL: crystal-pose LOO with this pipeline ----------
    rows_x = [{"y": d["y"], "length": d["length"], "feat": d["crystal"]} for d in data]
    sx = stat(loo(rows_x), yv)
    print("0. CONTROL — crystal oracle-pose LOO (this pipeline):")
    print(f"     r={sx[0]:+.3f} ρ={sx[1]:+.3f} RMSE={sx[2]:.2f}  (target ≈0.585 → {'REPRODUCES' if sx[0]>0.50 else 'DOES NOT reproduce → grader drift'})\n")

    # ---------- 1. AGGREGATIONS ----------
    def agg_rows(selector):
        out = []
        for d in data:
            feats = selector(d["poses"])
            mf = mean_feat(feats)
            if mf:
                out.append({"y": d["y"], "length": d["length"], "feat": mf})
        return out

    rank1 = agg_rows(lambda pf: [pf[0][2]])
    top5 = agg_rows(lambda pf: [x[2] for x in pf[:5]])
    mlbest1 = agg_rows(lambda pf: [min(pf, key=lambda x: x[1] if not np.isnan(x[1]) else 9e9)[2]])
    mlbest5 = agg_rows(lambda pf: [x[2] for x in sorted(pf, key=lambda x: x[1] if not np.isnan(x[1]) else 9e9)[:5]])

    print("1. AGGREGATIONS (production ridge + router, LOO):")
    for nm, rows in [("rank-1 (diffusion)", rank1), ("top-5 mean-feat", top5),
                     ("ML-best-1", mlbest1), ("ML-best-5 mean-feat", mlbest5)]:
        s = stat(loo(rows), np.array([r["y"] for r in rows]))
        print(f"     {nm:<22} r={s[0]:+.3f} ρ={s[1]:+.3f} RMSE={s[2]:.2f} (n={s[3]})")

    # (b) mean-predicted-ΔG: per test complex, fit on others' rank-1, predict & avg its top-5 poses
    predb = np.full(n, np.nan)
    base_rows = rank1
    for i in range(n):
        tr = [base_rows[j] for j in range(n) if j != i]
        params = fit_ridge(tr, PROD)
        ps = [predict(x[2], PROD, params) for x in data[i]["poses"][:5]]
        predb[i] = float(np.mean(ps))
    sb = stat(predb, yv)
    print(f"     {'(b) mean-pred-ΔG top5':<22} r={sb[0]:+.3f} ρ={sb[1]:+.3f} RMSE={sb[2]:.2f} (n={sb[3]})\n")

    # ---------- 2. PER-FEATURE DE-CORRELATION ----------
    print("2. PER-FEATURE DE-CORRELATION  Pearson(feat, y):  crystal → real(rank1) → real(top5)   Δ=real_top5−crystal")
    r1_feat = {r["feat"] is not None for r in rank1}  # noqa
    rows_for = {"crystal": rows_x, "rank1": rank1, "top5": top5}
    table = []
    for c in PROD:
        def colr(rows):
            x = np.array([r["feat"][c] for r in rows], float)
            yy = np.array([r["y"] for r in rows])
            m = ~np.isnan(x)
            return pearsonr(x[m], yy[m])[0] if m.sum() > 5 and np.std(x[m]) > 0 else np.nan
        rc, ra, rt = colr(rows_x), colr(rank1), colr(top5)
        table.append((c, rc, ra, rt, rt - rc))
    for c, rc, ra, rt, dl in sorted(table, key=lambda t: abs(t[4]) if t[4] == t[4] else 0, reverse=True):
        flag = "  <== BROKE" if (dl == dl and abs(dl) > 0.20) else ""
        print(f"     {c:<14} {rc:+.2f} → {ra:+.2f} → {rt:+.2f}   Δ={dl:+.2f}{flag}")

    # ---------- 3. POSE NOISE (within-complex CV across real poses) ----------
    print("\n3. POSE NOISE — within-complex coeff-of-variation across real poses (high = diluted signal):")
    cvs = {c: [] for c in PROD}
    for d in data:
        for c in PROD:
            v = np.array([x[2][c] for x in d["poses"] if x[2].get(c) is not None and not np.isnan(x[2][c])], float)
            if len(v) >= 3 and abs(v.mean()) > 1e-9:
                cvs[c].append(np.std(v) / (abs(v.mean()) + 1e-9))
    for c, v in sorted(cvs.items(), key=lambda kv: -np.nanmean(kv[1]) if kv[1] else 0):
        print(f"     {c:<14} CV={np.nanmean(v):.2f}")

    # ---------- 4. VALUE SHIFT crystal→real ----------
    print("\n4. VALUE SHIFT — (mean_real − mean_crystal) / σ_crystal  (systematic pose bias):")
    for c in PROD:
        xc = np.array([d["crystal"][c] for d in data if d["crystal"].get(c) is not None], float)
        xr = np.array([np.mean([x[2][c] for x in d["poses"] if x[2].get(c) is not None]) for d in data], float)
        sd = np.nanstd(xc) + 1e-9
        shift = (np.nanmean(xr) - np.nanmean(xc)) / sd
        flag = "  <== shifted" if abs(shift) > 0.5 else ""
        print(f"     {c:<14} {shift:+.2f}σ{flag}")

    # ---------- 5. LENGTH BREAKDOWN (best aggregation = top5) ----------
    print("\n5. LENGTH BREAKDOWN (top-5 aggregation, LOO):")
    pred_t5 = loo(top5)
    yt5 = np.array([r["y"] for r in top5])
    lens = np.array([r["length"] for r in top5])
    for nm, lo, hi in [("short ≤8", 0, 8), ("med 9–12", 9, 12), ("long 13–16", 13, 16), ("vlong ≥17", 17, 99)]:
        m = (lens >= lo) & (lens <= hi)
        if m.sum() >= 3:
            s = stat(pred_t5[m], yt5[m])
            print(f"     {nm:<12} n={m.sum():2d}  r={s[0]:+.3f}  RMSE={s[2]:.2f}  (y range {yt5[m].min():.1f}..{yt5[m].max():.1f})")
        else:
            print(f"     {nm:<12} n={m.sum():2d}  (too few)")

    # ---------- 6. RESIDUAL AUTOPSY ----------
    print("\n6. RESIDUAL AUTOPSY — corr(feature, |error|) on top-5 (what drives our error):")
    err = np.abs(yt5 - pred_t5)
    res_tab = []
    for c in PROD:
        x = np.array([r["feat"][c] for r in top5], float)
        m = ~(np.isnan(x) | np.isnan(err))
        if m.sum() > 5 and np.std(x[m]) > 0:
            res_tab.append((c, pearsonr(x[m], err[m])[0]))
    for c, rr in sorted(res_tab, key=lambda t: -abs(t[1]))[:6]:
        print(f"     {c:<14} corr(|err|)={rr:+.2f}")
    # pose quality vs error
    best_rmsd = np.array([min(x[0] for x in d["poses"]) for d in data])
    mean_rmsd = np.array([np.mean([x[0] for x in d["poses"][:5]]) for d in data])
    me = ~(np.isnan(best_rmsd) | np.isnan(err))
    print(f"     pose bestRMSD vs |err|  corr={pearsonr(best_rmsd[me], err[me])[0]:+.2f}  "
          f"(mean top5 RMSD={np.nanmean(mean_rmsd):.1f}Å, bestRMSD={np.nanmean(best_rmsd):.1f}Å)")
    print(f"     corr(bestRMSD, y)={pearsonr(best_rmsd[me], yt5[me])[0]:+.2f}  "
          f"(does pose quality track affinity at all?)")


if __name__ == "__main__":
    main()
