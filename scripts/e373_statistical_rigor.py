"""E373 — statistical rigor receipts answering external-review critiques (2026-07-10).

Three tests reviewers demanded, on the CURRENT headline model (16 STRUCT feats, 60%-id clustered CV, n=925/865):
  (1) BSA/length confound — partial correlation of pred vs exp controlling for buried-surface-area + length.
      (Critique: "cross-family ΔG is dominated by interface size; BSA alone gets 0.39; is your model a proxy?")
  (2) y-scramble permutation null — does the full pipeline reach the headline r on shuffled labels?
      (Critique: "300 experiments on a fixed set = in-search numbers, not out-of-sample.")
  (3) Steiger's Z for dependent correlations — is the ours-vs-clone head-to-head a real win or a tie?
      (Critique: "0.352 vs 0.325 with no significance test, rendered as a WIN arrow.")

Run: OMP_NUM_THREADS=1 LD_LIBRARY_PATH=$CONDA_PREFIX/lib python scripts/e373_statistical_rigor.py
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from scipy.stats import norm, pearsonr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e330_ours_pdbbind import FEATS, cluster_by_identity  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GB = lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0)


def _partial(a, b, ctrl):
    C = np.column_stack([ctrl, np.ones(len(a))])
    ra = a - C @ np.linalg.lstsq(C, a, rcond=None)[0]
    rb = b - C @ np.linalg.lstsq(C, b, rcond=None)[0]
    return pearsonr(ra, rb)[0]


def _steiger(rjk, rjh, rkh, n):
    rm2 = (rjk ** 2 + rjh ** 2) / 2
    f = (1 - rkh) / (2 * (1 - rm2)); h = (1 - f * rm2) / (1 - rm2)
    return (rjk - rjh) * np.sqrt((n - 3) / (2 * (1 - rkh) * h))


def main():
    rows = [json.loads(l) for l in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines() if l.strip()]
    X = np.nan_to_num(np.array([[float(r[k]) for k in FEATS] for r in rows], float))
    y = np.array([float(r["y"]) for r in rows]); seqs = [r["seq"] for r in rows]
    clu = cluster_by_identity(seqs, 0.60)
    L = np.array([len(s) for s in seqs]); BSA = X[:, FEATS.index("bsa_hyd")]
    pred = cross_val_predict(GB(), X, y, cv=GroupKFold(5), groups=clu)
    r = pearsonr(pred, y)[0]

    print(f"=== (1) BSA/length confound · n={len(rows)}, clustered CV ===")
    print(f"  corr(BSA, exp)={pearsonr(BSA, y)[0]:+.3f}   corr(length, exp)={pearsonr(L, y)[0]:+.3f}   corr(pred, exp)={r:+.3f}")
    print(f"  partial(pred, exp | BSA)        = {_partial(pred, y, BSA):+.3f}")
    print(f"  partial(pred, exp | length)     = {_partial(pred, y, L):+.3f}")
    print(f"  partial(pred, exp | BSA+length) = {_partial(pred, y, np.column_stack([BSA, L])):+.3f}   <<< signal is NOT a size proxy")

    print(f"\n=== (2) y-scramble permutation null (50x) ===")
    rng = np.random.default_rng(0)
    null = np.array([pearsonr((ps := cross_val_predict(GB(), X, (ys := rng.permutation(y)), cv=GroupKFold(5), groups=clu)), ys)[0] for _ in range(50)])
    print(f"  scrambled r: mean={null.mean():+.3f}  95th={np.percentile(null, 95):+.3f}  max={null.max():+.3f}   real r={r:.3f}  → p<{max(1, (null >= r).sum()) / len(null):.2f}")

    print(f"\n=== (3) Steiger's Z: ours vs PPI-clone head-to-head (matched 865) ===")
    ids = set(json.load(open(ROOT / "data/e331_matched_pdbids.json")))
    d3 = {json.loads(l)["pdb"]: json.loads(l) for l in (ROOT / "data/e180_protdcal3d.jsonl").read_text().splitlines() if l.strip()}
    m = [r_ for r_ in rows if r_["pdb"] in ids and r_["pdb"] in d3]
    ym = np.array([float(r_["y"]) for r_ in m]); clum = cluster_by_identity([r_["seq"] for r_ in m], 0.60)
    Xo = np.nan_to_num(np.array([[float(r_[k]) for k in FEATS] for r_ in m], float))
    Xc = np.nan_to_num(np.array([d3[r_["pdb"]]["desc"] for r_ in m], float))
    po = cross_val_predict(GB(), Xo, ym, cv=GroupKFold(5), groups=clum)
    pc = cross_val_predict(Pipeline([("s", StandardScaler()), ("m", SVR(C=4, gamma="scale"))]), Xc, ym, cv=GroupKFold(5), groups=clum)
    r1, r2, r12 = pearsonr(po, ym)[0], pearsonr(pc, ym)[0], pearsonr(po, pc)[0]
    z = _steiger(r1, r2, r12, len(ym)); p = 2 * (1 - norm.cdf(abs(z)))
    print(f"  ours r={r1:.3f}  clone r={r2:.3f}  Δ={r1 - r2:+.3f}  (pred-pred r={r12:.3f})")
    print(f"  Steiger Z={z:.2f}  p={p:.4f}  → {'SIGNIFICANT win' if p < 0.05 else 'NOT significant'}")


if __name__ == "__main__":
    main()
