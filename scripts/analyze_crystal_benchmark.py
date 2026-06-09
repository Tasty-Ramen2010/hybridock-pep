"""Analyze crystal-benchmark MM-GBSA results with size-confound controls.

The first 18 baseline points showed raw MM-GBSA ΔG is ~95% explained by peptide
length (a size meter, not an affinity predictor). This script evaluates each
scored variant under three models so we see the *residual binding* signal, not
the size artifact:

  * raw        — ΔG_pred vs ΔG_exp (the naive, size-confounded number).
  * per_residue— (ΔG_pred / n) vs ΔG_exp (cheapest size normalization).
  * +length    — LOO-CV ridge on [ΔG_pred, peptide_len] vs ΔG_exp (length carried
                 as an explicit covariate, so the fit explains affinity *beyond*
                 size — the honest measure of MM-GBSA's added value).

All correlations are sign-aware: we report |r| context against the Vina-docked
CV baseline of 0.42. Consumes whatever variant JSONs exist, so it runs on
partial results during the chain.

Usage:
    python scripts/analyze_crystal_benchmark.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]


def _loo_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    n = len(y)
    Xb = np.hstack([(X - X.mean(0)) / (X.std(0) + 1e-9), np.ones((n, 1))])
    reg = alpha * np.eye(Xb.shape[1]); reg[-1, -1] = 0.0
    pred = np.empty(n)
    for i in range(n):
        m = np.arange(n) != i
        c = np.linalg.solve(Xb[m].T @ Xb[m] + reg, Xb[m].T @ y[m])
        pred[i] = Xb[i] @ c
    return pred


def analyze(path: Path) -> None:
    rows = json.loads(path.read_text())
    rows = [r for r in rows if r.get("mmgbsa_dg") is not None and r.get("dg_exp") is not None]
    if len(rows) < 4:
        print(f"  {path.stem:30s} n={len(rows)} (too few)")
        return
    y = np.array([r["dg_exp"] for r in rows])
    p = np.array([r["mmgbsa_dg"] for r in rows])
    L = np.array([float(r.get("peptide_len") or 1) for r in rows])

    raw = pearsonr(p, y).statistic
    per_res = pearsonr(p / L, y).statistic
    cov = pearsonr(_loo_ridge(np.column_stack([p, L]), y), y).statistic
    conf = pearsonr(p, L).statistic  # size confound strength

    print(f"  {path.stem.replace('benchmark_crystal_scored_',''):10s} n={len(rows):2d} | "
          f"raw r={raw:+.3f}  per-res r={per_res:+.3f}  +length(CV) r={cov:+.3f} | "
          f"size-confound(ΔG~len)={conf:+.3f}")


def main() -> None:
    print("=== Crystal-benchmark MM-GBSA — size-controlled analysis ===")
    print("    (Vina-docked CV baseline to beat: r=0.42; sign: + = correct direction)\n")
    files = sorted(glob.glob(str(ROOT / "data" / "benchmark_crystal_scored_*.json")))
    if not files:
        # fall back to the single-output baseline name
        files = sorted(glob.glob(str(ROOT / "data" / "benchmark_crystal_scored.json")))
    for f in files:
        analyze(Path(f))
    print("\n  Reading: 'per-res' or '+length' beating |0.42| with the correct (+) sign")
    print("  is real MM-GBSA value beyond peptide size. 'raw' alone is a size meter.")


if __name__ == "__main__":
    main()
