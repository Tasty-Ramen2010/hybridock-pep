"""E99 — why do very-long peptides (≥17) score r=−0.515 on real poses? Test Ram's "floating" hypothesis.

DECISIVE TEST FIRST: crystal-pose LOO BY LENGTH BAND. If crystal-vlong is predictable but real-vlong
is not, the POSES are the defect (→ MD relaxation may help). If crystal-vlong ALSO fails, it's the
features/charged-floor/label-range (→ MD is irrelevant; don't waste GPU).

Then characterise the real-pose defect for vlong:
  * FLOATING metrics — interface coverage (fraction of peptide residues touching receptor), mean
    peptide→receptor separation, terminus lift-off. Crystal vs real, by band. "Floating" = low coverage.
  * EXTENDEDNESS — radius of gyration per residue, crystal vs real (the rg_per_L flip).
  * CHARGE — net / abs-charged fraction per band (charged-floor confounder for long peptides).
  * WITHIN-vlong correlations — each feature vs y, and vs |error|.
  * SIGNIFICANCE — bootstrap CI on the vlong r (n=15: is −0.515 real or noise?).
  * LEVER — pooled real r with vlong excluded / routed (ceiling from fixing vlong) vs PPI-Affinity 0.554.

Caches everything to runs/e99_cache.json so the MD pilot (e100) and further analysis reuse it.
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

from hybridock_pep.scoring.geometry_features import GEOMETRY_FEATURE_KEYS, compute_geometry_features  # noqa: E402
from hybridock_pep.scoring import pose_ranker_ml as PRM  # noqa: E402

CAMP = ROOT / "runs" / "e93_realpose_campaign"
CACHE = ROOT / "runs" / "e99_cache.json"
PROD = list(GEOMETRY_FEATURE_KEYS)
SHORT = ["bsa_hyd", "mj_contact", "strength_bur"]
P = PDBParser(QUIET=True)
NPOSE = 25
POS = "KR"
NEG = "DE"


def heavy_xyz(pdb):
    xyz, res = [], []
    for ln in Path(pdb).read_text().splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        an = ln[12:16].strip()
        if not an or an[0] in ("H", "D"):
            continue
        try:
            xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
            res.append((ln[21], ln[22:26]))
        except ValueError:
            continue
    return np.array(xyz) if xyz else np.empty((0, 3)), res


def ca_coords(pdb):
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


def floating(pep_pdb, rec_xyz):
    """interface coverage (frac peptide residues w/ heavy atom <4.5Å of receptor), mean min-dist, rg."""
    pxyz, pres = heavy_xyz(pep_pdb)
    if len(pxyz) == 0 or len(rec_xyz) == 0:
        return None
    d = np.sqrt(((pxyz[:, None, :] - rec_xyz[None, :, :]) ** 2).sum(2)).min(1)  # per peptide-atom min dist
    # per residue: min over its atoms
    byres = {}
    for dd, r in zip(d, pres):
        byres.setdefault(r, []).append(dd)
    rmins = np.array([min(v) for v in byres.values()])
    cov = float((rmins < 4.5).mean())
    ca = ca_coords(pep_pdb)
    rg = float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(1).mean())) if len(ca) >= 3 else np.nan
    return cov, float(rmins.mean()), float(np.median(d)), rg, len(byres)


def build_cache():
    bench = {r["pdb"]: r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    bundle = joblib.load(PRM.DEFAULT_MODEL_PATH)
    phi_kde, psi_kde, ml = bundle["phi_kde"], bundle["psi_kde"], bundle["model"]
    complexes = sorted([d.name for d in CAMP.iterdir() if (d / "poses").exists()])
    out = []
    for cx in complexes:
        meta = bench.get(cx)
        if not meta or meta.get("dg_exp") is None:
            continue
        rec = Path(meta["pocket_pdb"]).resolve()
        rec_xyz, _ = heavy_xyz(rec)
        seq = meta.get("peptide_seq", "")
        L = int(meta.get("peptide_len") or len(seq))
        y = float(meta["dg_exp"])
        xtal = Path(meta["peptide_pdb"])
        xca = ca_coords(xtal)
        cf = compute_geometry_features(xtal, rec)
        cfl = floating(xtal, rec_xyz)
        poses = sorted((CAMP / cx / "poses").glob("pose_*.pdb"), key=lambda q: int(q.stem.split("_")[1]))[:NPOSE]
        pl = []
        for p in poses:
            f = compute_geometry_features(p, rec)
            if not f:
                continue
            mlf = PRM.compute_features(p, phi_kde, psi_kde)
            ms = float(ml.predict(np.array([mlf]))[0]) if mlf else None
            fl = floating(p, rec_xyz)
            pl.append({"feat": f, "rmsd": rmsd(ca_coords(p), xca), "ml": ms,
                       "cov": fl[0] if fl else None, "sep": fl[1] if fl else None, "rg": fl[3] if fl else None})
        if cf is None or not pl:
            continue
        netc = sum(seq.count(a) for a in POS) - sum(seq.count(a) for a in NEG)
        absc = (sum(seq.count(a) for a in POS + NEG)) / max(1, L)
        out.append({"pdb": cx, "y": y, "length": L, "seq": seq, "net_charge": netc, "abs_charge_frac": absc,
                    "crystal": cf, "crystal_cov": cfl[0] if cfl else None, "crystal_rg": cfl[3] if cfl else None,
                    "poses": pl})
        print(f"  {cx}: L={L} y={y:+.2f} net={netc:+d} xtal_cov={cfl[0] if cfl else 0:.2f} poses={len(pl)}", flush=True)
    CACHE.write_text(json.dumps(out))
    return out


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
        if router and rows[i]["length"] <= 8 and sum(r["length"] <= 8 for r in tr) >= 6:
            cols, base = SHORT, [r for r in tr if r["length"] <= 8]
        else:
            cols, base = PROD, tr
        pred[i] = predict(rows[i]["feat"], cols, fit_ridge(base, cols))
    return pred


def st(p, y):
    m = ~(np.isnan(p) | np.isnan(y))
    if m.sum() < 4:
        return (np.nan, np.nan, int(m.sum()))
    return (pearsonr(p[m], y[m])[0], float(np.sqrt(np.mean((p[m] - y[m]) ** 2))), int(m.sum()))


def mean_feat(feats):
    return {k: float(np.nanmean([f["feat"][k] for f in feats if f["feat"].get(k) is not None])) for k in PROD}


def bands(L):
    return ("short≤8" if L <= 8 else "med9-12" if L <= 12 else "long13-16" if L <= 16 else "vlong≥17")


def main():
    if CACHE.exists() and "--rebuild" not in sys.argv:
        data = json.loads(CACHE.read_text())
        print(f"=== E99 (cached {len(data)} complexes) ===\n")
    else:
        print("=== E99 building cache ===")
        data = build_cache()
        print(f"\ncached {len(data)} complexes\n")

    # rows: rank-1, top5 mean, ML-best5 mean
    def rows_for(sel):
        r = []
        for d in data:
            fs = sel(d["poses"])
            if fs:
                r.append({"pdb": d["pdb"], "y": d["y"], "length": d["length"], "feat": mean_feat(fs)})
        return r
    rank1 = rows_for(lambda pl: [pl[0]])
    top5 = rows_for(lambda pl: pl[:5])
    mlbest5 = rows_for(lambda pl: sorted([x for x in pl if x["ml"] is not None], key=lambda x: x["ml"])[:5] or pl[:5])
    crystal = [{"pdb": d["pdb"], "y": d["y"], "length": d["length"], "feat": d["crystal"]} for d in data]
    yv = np.array([d["y"] for d in data])

    print("=== A. DECISIVE: LOO r BY LENGTH BAND — crystal(oracle) vs real ===")
    print(f"{'band':<10}{'n':>4}{'crystal r':>12}{'real top5 r':>14}{'real ML5 r':>13}   verdict")
    pc, p5, pm = loo(crystal), loo(top5), loo(mlbest5)
    Lc = np.array([r["length"] for r in crystal])
    for bnd in ["short≤8", "med9-12", "long13-16", "vlong≥17"]:
        m = np.array([bands(L) == bnd for L in Lc])
        if m.sum() < 3:
            print(f"{bnd:<10}{m.sum():>4}   (too few)")
            continue
        rc = st(pc[m], yv[m])[0]
        r5 = st(p5[m], yv[m])[0]
        rm = st(pm[m], yv[m])[0]
        verdict = ""
        if bnd == "vlong≥17":
            verdict = "POSES are the defect → MD may help" if (rc > 0.2 and r5 < 0.1) else \
                      "crystal ALSO fails → features/floor, MD won't help" if rc < 0.2 else "mixed"
        print(f"{bnd:<10}{m.sum():>4}{rc:>+12.3f}{r5:>+14.3f}{rm:>+13.3f}   {verdict}")

    # ---- vlong focus ----
    vl = [d for d in data if d["length"] >= 17]
    print(f"\n=== B. FLOATING metrics (vlong n={len(vl)}) — crystal vs real (low coverage = floating) ===")
    for bnd in ["med9-12", "long13-16", "vlong≥17"]:
        sub = [d for d in data if bands(d["length"]) == bnd]
        if not sub:
            continue
        xcov = np.nanmean([d["crystal_cov"] for d in sub])
        rcov = np.nanmean([np.nanmean([p["cov"] for p in d["poses"] if p["cov"] is not None]) for d in sub])
        rsep = np.nanmean([np.nanmean([p["sep"] for p in d["poses"] if p["sep"] is not None]) for d in sub])
        xrg = np.nanmean([d["crystal_rg"] for d in sub])
        rrg = np.nanmean([np.nanmean([p["rg"] for p in d["poses"] if p["rg"] is not None]) for d in sub])
        print(f"  {bnd:<10} interface-cov crystal={xcov:.2f} real={rcov:.2f} (Δ={rcov-xcov:+.2f})  "
              f"mean-sep={rsep:.1f}Å  rg crystal={xrg:.1f} real={rrg:.1f} (Δ={rrg-xrg:+.1f}Å)")

    print(f"\n=== C. WITHIN-vlong correlations (n={len(vl)}) — feature vs y (real top5) ===")
    vrows = [r for r in top5 if r["length"] >= 17]
    vy = np.array([r["y"] for r in vrows])
    cors = []
    for c in PROD:
        x = np.array([r["feat"][c] for r in vrows], float)
        m = ~np.isnan(x)
        if m.sum() > 4 and np.std(x[m]) > 0:
            cors.append((c, pearsonr(x[m], vy[m])[0]))
    for c, r in sorted(cors, key=lambda t: -abs(t[1]))[:8]:
        print(f"     {c:<14} r(feat,y)={r:+.2f}")
    nc = np.array([d["net_charge"] for d in vl], float)
    ac = np.array([d["abs_charge_frac"] for d in vl], float)
    print(f"     net_charge   r(,y)={pearsonr(nc, np.array([d['y'] for d in vl]))[0]:+.2f}   "
          f"abs_charge_frac r(,y)={pearsonr(ac, np.array([d['y'] for d in vl]))[0]:+.2f}")

    print(f"\n=== D. SIGNIFICANCE of vlong real r (n={len(vrows)}) — bootstrap 2000x ===")
    pv = p5[np.array([bands(L) == 'vlong≥17' for L in Lc])]
    yv2 = yv[np.array([bands(L) == 'vlong≥17' for L in Lc])]
    mm = ~(np.isnan(pv) | np.isnan(yv2))
    pv, yv2 = pv[mm], yv2[mm]
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        idx = rng.integers(0, len(pv), len(pv))
        if np.std(pv[idx]) > 0 and np.std(yv2[idx]) > 0:
            bs.append(pearsonr(pv[idx], yv2[idx])[0])
    bs = np.array(bs)
    print(f"     real vlong r={pearsonr(pv, yv2)[0]:+.3f}  95% CI=[{np.percentile(bs,2.5):+.2f}, {np.percentile(bs,97.5):+.2f}]  "
          f"P(r<0)={(bs<0).mean():.2f}  (y-range={yv2.max()-yv2.min():.1f} kcal)")

    print("\n=== E. LEVER — pooled real r if vlong fixed (vs PPI-Affinity 0.554) ===")
    keep = np.array([L < 17 for L in Lc])
    print(f"     real top5 ALL          r={st(p5, yv)[0]:+.3f}")
    print(f"     real top5 EXCL vlong   r={st(p5[keep], yv[keep])[0]:+.3f}   (ceiling if vlong routed out)")
    print(f"     real ML5  ALL          r={st(pm, yv)[0]:+.3f}")
    print(f"     real ML5  EXCL vlong   r={st(pm[keep], yv[keep])[0]:+.3f}")


if __name__ == "__main__":
    main()
