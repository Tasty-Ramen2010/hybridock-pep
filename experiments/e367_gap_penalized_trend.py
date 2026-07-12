"""E367 — does the short-peptide identity artifact distort the leakage trend? (Ram's catch)

The e330/e366 identity is a GLOBAL alignment with match=1, mismatch=0, and FREE gaps → it equals
longest-common-subsequence / shorter-length. For short peptides that massively over-merges: GGA vs ACC
scores 0.33 from a single shared residue placed by a free gap, and GGA vs CGG (shifted) scores 0.67.
This re-runs the identity-threshold sweep with a GAP-PENALISED aligner (open −1, extend −0.5) that respects
residue placement, and compares clusters + MAE/RMSE/r to the free-gap metric.

Run: OMP_NUM_THREADS=1 LD_LIBRARY_PATH=$CONDA_PREFIX/lib python experiments/e367_gap_penalized_trend.py
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e330_ours_pdbbind import FEATS, metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]


def make_aligner(gap_penalised: bool):
    from Bio.Align import PairwiseAligner
    a = PairwiseAligner()
    a.mode = "global"
    a.match_score, a.mismatch_score = 1.0, 0.0
    a.open_gap_score = (-1.0 if gap_penalised else 0.0)
    a.extend_gap_score = (-0.5 if gap_penalised else 0.0)
    return a


def cluster(seqs, thresh, aligner):
    """Identical to e330.cluster_by_identity (incl. its length guard); ONLY the aligner's gaps differ."""
    uniq = sorted(set(seqs), key=lambda s: (-len(s), s))
    reps: list[str] = []
    rep_of: dict[str, int] = {}
    for s in uniq:
        placed = False
        for ci, rep in enumerate(reps):
            if min(len(s), len(rep)) < thresh * max(len(s), len(rep)):
                continue  # e330 length guard: lengths too different to reach thresh
            ident = 1.0 if s == rep else max(0.0, aligner.score(s, rep)) / min(len(s), len(rep))
            if ident >= thresh:
                rep_of[s] = ci
                placed = True
                break
        if not placed:
            rep_of[s] = len(reps)
            reps.append(s)
    return np.array([rep_of[s] for s in seqs])


def main():
    rows = [json.loads(l) for l in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines() if l.strip()]
    X = np.nan_to_num(np.array([[float(r[k]) for k in FEATS] for r in rows], float))
    y = np.array([float(r["y"]) for r in rows])
    seqs = [r["seq"] for r in rows]
    mk = lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                           subsample=0.8, random_state=0)

    print(f"=== free-gap (current)  vs  gap-penalised (placement-aware) identity · {len(rows)} complexes ===\n")
    print(f"  {'cutoff':>7} | {'clusters':>8} {'MAE':>5} {'RMSE':>5} {'r':>7}  (free-gap) | "
          f"{'clusters':>8} {'MAE':>5} {'RMSE':>5} {'r':>7}  (gap-penalised)")
    print("  " + "-" * 96)
    free_al, gap_al = make_aligner(False), make_aligner(True)
    for th in sorted(THRESHOLDS, reverse=True):
        line = f"  {int(th*100):>6}% |"
        for al in (free_al, gap_al):
            cl = cluster(seqs, th, al)
            nc = len(set(cl.tolist()))
            oof = cross_val_predict(mk(), X, y, cv=GroupKFold(min(5, nc)), groups=cl)
            r, sp, rmse, mae = metrics(y, oof)
            line += f" {nc:>8} {mae:>5.2f} {rmse:>5.2f} {r:>+7.3f} |"
        print(line)
    print("\n  If the gap-penalised columns show MORE clusters + steadier r, the free-gap metric was\n"
          "  over-merging unrelated short peptides — i.e. our clustered split was, if anything, TOO loose\n"
          "  at low cutoffs (spurious near-twins), not too strict.")


if __name__ == "__main__":
    main()
