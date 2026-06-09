"""Phase 0 scoring-evaluation harness.

Frozen, reproducible evaluation of any ΔG predictor against the holdout set,
with an honest Kd+Ki-only "clean" subset (IC50/EC50 dropped — they are
assay-dependent, not equilibrium constants, and inject 1-3 kcal/mol of noise).

Metrics reported per predictor:
  * Pearson r   — primary (discriminative power; the honest limiting metric).
  * Spearman rho — rank correlation.
  * MAE / RMSE  — kcal/mol (interpret against the mean-predictor baseline,
    which is RMSE = std(ΔG); on peptides the dynamic range is narrow so a
    low RMSE alone is NOT evidence of skill).

Usage:
    python scripts/eval_scoring.py                 # baseline table
    python scripts/eval_scoring.py --write-clean   # also write clean subset
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = ROOT / "data" / "eval_holdout_calibrations.json"
CLEAN_OUT = ROOT / "data" / "eval_kd_ki_clean.json"

# Equilibrium-constant affinity types only. IC50/EC50 are assay-dependent.
_CLEAN_TYPES = {"Kd", "Ki"}


def load_rows(path: Path = HOLDOUT) -> list[dict]:
    rows = json.loads(path.read_text())
    return [r for r in rows if r.get("vina") is not None and r.get("dg_exp") is not None]


def clean_subset(rows: list[dict]) -> list[dict]:
    """Kd+Ki only — the honest evaluation set."""
    return [r for r in rows if r.get("affinity_type") in _CLEAN_TYPES]


def _arr(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(r.get(key, 0) or 0) for r in rows], dtype=float)


def metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "pearson_r": float(pearsonr(pred, y).statistic),
        "spearman_rho": float(spearmanr(pred, y).statistic),
        "mae": float(np.mean(np.abs(pred - y))),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
    }


def refit_ceiling(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """In-sample slope+intercept refit of a single feature onto ΔG — the
    optimistic ceiling for that feature alone (NOT held out; report honestly)."""
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return m * x + b


def evaluate(rows: list[dict], label: str) -> None:
    y = _arr(rows, "dg_exp")
    v = _arr(rows, "vina")
    ss = _arr(rows, "s_ss_weighted")
    print(f"\n=== {label} (n={len(rows)}) ===")
    print(f"  ΔG_exp mean={y.mean():.2f} std={y.std():.2f}  "
          f"(mean-predictor RMSE = {y.std():.2f} kcal/mol — the bar to beat)")
    predictors = {
        "vina_raw": v,
        "v1.4_live": 1.0 * v - 0.3379828 * ss + 1.5367,
        "v1.2": -0.4341 * ss - 3.9462,
        "vina_refit(ceil)": refit_ceiling(v, y),
    }
    print(f"  {'predictor':20s} {'r':>7s} {'rho':>7s} {'MAE':>6s} {'RMSE':>6s}")
    for name, pred in predictors.items():
        m = metrics(pred, y)
        print(f"  {name:20s} {m['pearson_r']:+7.3f} {m['spearman_rho']:+7.3f} "
              f"{m['mae']:6.2f} {m['rmse']:6.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-clean", action="store_true",
                    help="Write the Kd+Ki clean subset to data/eval_kd_ki_clean.json")
    args = ap.parse_args()

    rows = load_rows()
    clean = clean_subset(rows)
    evaluate(rows, "ALL (mixed affinity types)")
    evaluate(clean, "CLEAN (Kd+Ki only)")

    by_type: dict[str, int] = {}
    for r in rows:
        by_type[r.get("affinity_type", "?")] = by_type.get(r.get("affinity_type", "?"), 0) + 1
    print(f"\naffinity-type counts: {by_type}")

    if args.write_clean:
        CLEAN_OUT.write_text(json.dumps(clean, indent=2))
        print(f"\nWrote {len(clean)} Kd+Ki complexes → {CLEAN_OUT}")


if __name__ == "__main__":
    main()
