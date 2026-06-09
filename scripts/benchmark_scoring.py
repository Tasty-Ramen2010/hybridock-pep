"""Scoring benchmark harness — the measurement foundation (overhaul step 1).

The single source of truth for "how good is our scoring function." Unlike
scripts/eval_scoring.py (which reports an optimistic in-sample slope+intercept
*ceiling*), this evaluates every model with **leave-one-out cross-validation**,
so the reported Pearson r is an honest generalization estimate — the number we
must move to claim progress toward MM/PBSA SOTA.

Design:
  * Reads a manifest of complexes with per-complex features. Today that is the
    clean Kd+Ki set (data/eval_kd_ki_clean.json); the same code consumes any
    JSON list carrying the feature keys, so MM-GBSA / IE columns slot in later.
  * Evaluates named feature models (vina-only, AD4, entropy variants, and any
    ridge combination) under LOO-CV.
  * Reports CV Pearson r, Spearman, RMSE, plus the mean-predictor RMSE bar.
  * Optional --by-family for per-cluster breakdown when a 'family' key exists.

Usage:
    python scripts/benchmark_scoring.py
    python scripts/benchmark_scoring.py --manifest data/eval_kd_ki_clean.json
    python scripts/benchmark_scoring.py --features vina_score s_ss_weighted n_contact
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "eval_kd_ki_clean.json"

# Candidate single-feature and combination models. Each maps a label to the
# list of feature keys fed to a LOO-CV ridge fit against ΔG_exp.
_MODELS: dict[str, list[str]] = {
    "vina_only": ["vina_score"],
    "ad4_only": ["ad4_score"],
    "entropy_only": ["s_ss_weighted"],
    "vina+entropy": ["vina_score", "s_ss_weighted"],
    "vina+ad4": ["vina_score", "ad4_score"],
    "vina+ncontact": ["vina_score", "n_contact"],
    "vina+entropy+ncontact": ["vina_score", "s_ss_weighted", "n_contact"],
    "all_features": ["vina_score", "ad4_score", "n_contact", "s_ss_weighted",
                     "s_sc_sum", "s_bb_sum"],
}


def load_manifest(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    return [r for r in rows if r.get("dg_exp") is not None and r.get("vina") is not None]


def _feature_matrix(rows: list[dict], keys: list[str]) -> np.ndarray | None:
    """Build (n, k) feature matrix; returns None if any key is entirely absent."""
    cols = []
    for k in keys:
        # tolerate both 'vina' and 'vina_score' style keys
        alias = {"vina_score": "vina", "ad4_score": "ad4", "n_contact": "n_contact"}.get(k, k)
        vals = [r.get(k, r.get(alias)) for r in rows]
        if all(v is None for v in vals):
            return None
        cols.append([float(v or 0.0) for v in vals])
    return np.column_stack(cols)


def loo_cv(X: np.ndarray, y: np.ndarray, ridge_alpha: float = 1.0) -> np.ndarray:
    """Leave-one-out CV predictions via ridge regression (numpy closed form)."""
    n = len(y)
    preds = np.empty(n)
    Xb = np.hstack([X, np.ones((n, 1))])
    k = Xb.shape[1]
    reg = ridge_alpha * np.eye(k)
    reg[-1, -1] = 0.0  # don't regularize intercept
    for i in range(n):
        mask = np.arange(n) != i
        Xi, yi = Xb[mask], y[mask]
        coef = np.linalg.solve(Xi.T @ Xi + reg, Xi.T @ yi)
        preds[i] = Xb[i] @ coef
    return preds


def evaluate(rows: list[dict], features: list[str] | None) -> None:
    y = np.array([float(r["dg_exp"]) for r in rows])
    print(f"\n=== Scoring benchmark (LOO-CV) — n={len(rows)} ===")
    print(f"  ΔG_exp mean={y.mean():.2f} std={y.std():.2f} "
          f"(mean-predictor RMSE = {y.std():.2f} — the bar)")
    print(f"  {'model':24s} {'CV r':>7s} {'CV rho':>7s} {'CV RMSE':>8s}")

    models = {"custom": features} if features else _MODELS
    for label, keys in models.items():
        X = _feature_matrix(rows, keys)
        if X is None:
            print(f"  {label:24s}   (feature missing — skipped: {keys})")
            continue
        # Standardize features so ridge penalty is even-handed.
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        pred = loo_cv(Xs, y)
        r = pearsonr(pred, y).statistic
        rho = spearmanr(pred, y).statistic
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        print(f"  {label:24s} {r:+7.3f} {rho:+7.3f} {rmse:8.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--features", nargs="+", default=None,
                    help="Custom feature key list to evaluate as one model.")
    args = ap.parse_args()
    rows = load_manifest(args.manifest)
    evaluate(rows, args.features)


if __name__ == "__main__":
    main()
