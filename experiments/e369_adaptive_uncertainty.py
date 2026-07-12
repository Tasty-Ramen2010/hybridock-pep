"""E369 — can we give EASY predictions a tight band (±0.8) and HARD ones a wide band (±3) — honestly?

Adaptive/heteroscedastic intervals are only valid if some per-prediction signal σ(x) actually correlates with
|error|. Length/charge did not (E-check R²=0.003). Here we test stronger candidates and, if one works, build a
Mondrian conformal predictor (separate calibrated width per difficulty bin) and report per-bin width + coverage.

Signals tested (all leakage-safe: unsupervised, or out-of-fold):
  A. kNN distance in standardized feature space   (sparse feature region → harder)   [unsupervised]
  B. GBT trained OUT-OF-FOLD to predict |residual| from the 16 features               [supervised, OOF]
  C. |pred − median(pred)|  (prediction extremity)

Run: OMP_NUM_THREADS=1 LD_LIBRARY_PATH=$CONDA_PREFIX/lib python experiments/e369_adaptive_uncertainty.py
"""
from __future__ import annotations
import csv, json, math, os, sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e330_ours_pdbbind import FEATS, cluster_by_identity  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def q_conf(res, cov):
    n = len(res); k = math.ceil((n + 1) * cov)
    return float(np.max(res)) if k > n else float(np.sort(res)[k - 1])


def main():
    feat = {json.loads(l)["pdb"].upper(): json.loads(l)
            for l in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines() if l.strip()}
    rows = [r for r in csv.DictReader(open(ROOT / "data/hybridock_blind_925.csv")) if r["pdb"].upper() in feat]
    X = np.nan_to_num(np.array([[float(feat[r["pdb"].upper()][k]) for k in FEATS] for r in rows], float))
    Xs = StandardScaler().fit_transform(X)
    p = np.array([float(r["pred_dG_kcal_mol"]) for r in rows])
    y = np.array([float(r["exp_dG_kcal_mol"]) for r in rows])
    res = np.abs(p - y)
    seqs = [r["peptide"] for r in rows]
    clu = cluster_by_identity(seqs, 0.60)

    # ---- candidate difficulty signals ----
    nn = NearestNeighbors(n_neighbors=6).fit(Xs)
    knn = nn.kneighbors(Xs)[0][:, 1:].mean(axis=1)                    # A: mean dist to 5 NN
    gbt = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0)
    sig_gbt = cross_val_predict(gbt, X, res, cv=GroupKFold(5), groups=clu)   # B: OOF predicted |resid|
    extremity = np.abs(p - np.median(p))                             # C

    print(f"n={len(rows)}  MAE={res.mean():.2f}  RMSE={np.sqrt((res**2).mean()):.2f}\n")
    print("=== does any signal correlate with |error|? (need this > ~0 for adaptive to be honest) ===")
    cands = {"A: kNN feature-distance": knn, "B: OOF |resid| model": sig_gbt, "C: |pred−median|": extremity}
    best, bestr = None, 0.0
    for name, s in cands.items():
        r = pearsonr(s, res)[0]; rho = spearmanr(s, res).statistic
        # decile separation: mean|err| in lowest vs highest σ decile
        o = np.argsort(s); lo = res[o[: len(o)//10]].mean(); hi = res[o[-len(o)//10:]].mean()
        print(f"  {name:26s} pearson={r:+.3f} spearman={rho:+.3f}   easy-decile |err|={lo:.2f}  hard-decile |err|={hi:.2f}")
        if abs(r) > abs(bestr):
            best, bestr = (name, s), r

    name, s = best
    print(f"\nbest signal: {name}  (pearson {bestr:+.3f})")
    if abs(bestr) < 0.15:
        print("\n>>> VERDICT: no signal meaningfully predicts our error. Adaptive ±0.8/±3 bands would be FAKE —\n"
              "    they'd vary width without tracking real accuracy, so 'confident' ones wouldn't actually be\n"
              "    more correct. The honest product is the GLOBAL conformal interval (E368). Not shippable as adaptive.")
        return

    # ---- Mondrian conformal: per-difficulty-bin width, each calibrated to valid coverage ----
    print(f"\n=== Mondrian conformal by {name.split(':')[0]} quartiles — 80% target, cluster-split test (200x) ===")
    order = np.argsort(s)
    binid = np.empty(len(s), int)
    for b, idx in enumerate(np.array_split(order, 4)):
        binid[idx] = b
    uclu = np.array(sorted(set(clu.tolist()))); rng = np.random.default_rng(0)
    cov = {b: [] for b in range(4)}; wid = {b: [] for b in range(4)}
    for _ in range(200):
        rng.shuffle(uclu); cal_cl = set(uclu[:len(uclu)//2].tolist())
        cal = np.array([c in cal_cl for c in clu])
        for b in range(4):
            m_cal = cal & (binid == b); m_te = (~cal) & (binid == b)
            if m_cal.sum() < 10 or m_te.sum() < 5:
                continue
            qb = q_conf(res[m_cal], 0.80)
            cov[b].append(float((res[m_te] <= qb).mean())); wid[b].append(qb)
    print(f"  {'bin':>16} {'80%-coverage':>13} {'half-width':>12} {'mean|err|':>10}")
    for b in range(4):
        lab = ["easiest Q1", "Q2", "Q3", "hardest Q4"][b]
        m = binid == b
        print(f"  {lab:>16} {np.mean(cov[b])*100:>11.1f}% {'':2}±{np.mean(wid[b]):>4.2f} kcal {res[m].mean():>9.2f}")
    print("\n  If Q1 width << Q4 width AND every bin holds ~80% coverage → honest adaptive bands. If widths are\n"
          "  all similar, the signal is too weak and global is the honest choice.")


if __name__ == "__main__":
    main()
