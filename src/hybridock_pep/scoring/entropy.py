"""Backbone entropy correction and hybrid score calibration (SCORE-03).

Implements the D-01 hybrid score formula:
    hybrid = vina + beta*(ad4 - vina) + alpha*n_residues

where ``alpha * n_residues`` is the backbone entropy correction term
and ``beta`` controls the blending weight of AD4 relative to Vina.

Calibration (alpha, beta) is loaded from a JSON file and validated
on every read per T-03-09 (load_calibration raises ValueError for
out-of-range values). Fitting uses scipy L-BFGS-B with hardcoded bounds.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr

from hybridock_pep.models import ScoredPose

_log = logging.getLogger(__name__)

# Thermodynamic constant: RT at 298 K in kcal/mol (D-09, hardcoded in v1).
_RT = 0.592


def load_calibration(path: Path) -> dict:
    """Load and validate calibration parameters from a JSON file.

    Validates that alpha is within the physiologically meaningful range
    [0.2, 1.2] kcal/mol/residue and that beta is within [0.0, 0.5].
    Values outside these ranges indicate a broken training set or
    misconfigured pipeline (CLAUDE.md §9; SCORE-03 abort).

    Args:
        path: Path to the calibration JSON file (D-11 schema).

    Returns:
        Dictionary containing calibration data including 'alpha' and 'beta'.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If alpha is outside [0.2, 1.2] or beta is outside [0.0, 0.5],
            with a diagnostic message quoting the bad value and valid range.
    """
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    with path.open() as fh:
        cal = json.load(fh)

    try:
        alpha = cal["alpha"]
        beta = cal["beta"]
    except KeyError as exc:
        raise ValueError(
            f"Calibration file {path} is missing required key {exc}. "
            "Re-run calibrate_alpha.py to regenerate a valid calibration file."
        ) from exc

    if not (0.2 <= alpha <= 1.2):
        raise ValueError(
            f"Calibrated α={alpha:.3f} is outside valid range [0.2, 1.2] kcal/mol/residue "
            "— check training data coverage. SCORE-03 abort."
        )
    if not (0.0 <= beta <= 0.5):
        raise ValueError(
            f"Calibrated β={beta:.3f} is outside valid range [0.0, 0.5] "
            "— β > 0.5 means AD4 dominates over Vina, contradicting the Vina-primary design. "
            "Check training data or use default calibration.json."
        )

    _log.debug("Loaded calibration: alpha=%.3f beta=%.3f from %s", alpha, beta, path)
    return cal


def write_calibration(
    path: Path,
    alpha: float,
    beta: float,
    **kwargs: float | int | str,
) -> None:
    """Write calibration parameters to a JSON file (D-11 schema).

    Always sets 'calibrated_at' to the current UTC time in ISO 8601 format.
    Creates parent directories as needed.

    Args:
        path: Destination path for the calibration JSON file.
        alpha: Backbone entropy coefficient (kcal/mol/residue).
        beta: AD4 blending weight (dimensionless, [0.0, 0.5]).
        **kwargs: Additional D-11 fields to include (e.g., n_complexes,
            pearson_r, rmse_kcal_mol, training_csv).
    """
    payload = {
        "alpha": alpha,
        "beta": beta,
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
        **kwargs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    _log.info("Wrote calibration to %s (alpha=%.3f, beta=%.3f)", path, alpha, beta)


def apply_hybrid_score(
    pose: ScoredPose,
    *,
    alpha: float,
    beta: float,
    n_residues: int,
) -> None:
    """Apply the D-01 hybrid score formula to a ScoredPose in place.

    Sets ``pose.entropy_correction = alpha * n_residues`` and
    ``pose.hybrid_score = vina + beta*(ad4 - vina) + alpha*n_residues``.

    Note:
        This function does NOT validate alpha or beta ranges. Range
        validation is the responsibility of load_calibration() (T-03-09).
        Callers must ensure vina_score and ad4_score are populated before
        calling this function (T-03-11).

    Args:
        pose: ScoredPose with vina_score and ad4_score already set.
        alpha: Backbone entropy coefficient (kcal/mol/residue).
        beta: AD4 blending weight (dimensionless).
        n_residues: Number of residues in the peptide.

    Raises:
        AssertionError: If pose.vina_score or pose.ad4_score is None.
    """
    assert pose.vina_score is not None, "vina_score must be set before apply_hybrid_score"
    assert pose.ad4_score is not None, "ad4_score must be set before apply_hybrid_score"

    pose.entropy_correction = alpha * n_residues
    pose.hybrid_score = (
        pose.vina_score + beta * (pose.ad4_score - pose.vina_score) + pose.entropy_correction
    )
    _log.debug(
        "Pose %d: vina=%.3f ad4=%.3f ec=%.3f hybrid=%.3f",
        pose.pose_idx,
        pose.vina_score,
        pose.ad4_score,
        pose.entropy_correction,
        pose.hybrid_score,
    )


def fit_calibration(
    vina_scores: list[float],
    ad4_scores: list[float],
    n_residues_list: list[int],
    experimental_pkd: list[float],
) -> dict:
    """Fit calibration parameters (alpha, beta) using L-BFGS-B minimization.

    Minimises sum-of-squared residuals between hybrid scores and experimental
    ΔG values converted via D-09: ΔG = -RT * pKd, where RT = 0.592 kcal/mol
    at 298 K (hardcoded, not a CLI parameter in v1).

    Bounds are enforced by scipy L-BFGS-B:
        alpha ∈ [0.2, 1.2] kcal/mol/residue
        beta  ∈ [0.0, 0.5]

    Starting point: x0 = [0.65, 0.22] (D-10 defaults).

    Args:
        vina_scores: List of Vina --score_only values in kcal/mol.
        ad4_scores: List of AutoDock4 scoring values in kcal/mol.
        n_residues_list: List of peptide lengths (number of residues).
        experimental_pkd: List of experimental pKd values.

    Returns:
        Dictionary with keys: 'alpha', 'beta', 'pearson_r', 'rmse_kcal_mol'.

    Raises:
        ValueError: If input lists have different lengths.
    """
    n = len(vina_scores)
    if not (n == len(ad4_scores) == len(n_residues_list) == len(experimental_pkd)):
        raise ValueError(
            "All input lists must have the same length; got lengths: "
            f"vina={len(vina_scores)}, ad4={len(ad4_scores)}, "
            f"n_res={len(n_residues_list)}, pkd={len(experimental_pkd)}"
        )

    delta_g = [-_RT * pkd for pkd in experimental_pkd]

    def objective(params: np.ndarray) -> float:
        alpha, beta = params
        residuals = [
            (v + beta * (a - v) + alpha * nr) - dg
            for v, a, nr, dg in zip(vina_scores, ad4_scores, n_residues_list, delta_g)
        ]
        return float(sum(r**2 for r in residuals))

    x0 = np.array([0.65, 0.22])
    bounds = [(0.2, 1.2), (0.0, 0.5)]
    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        _log.warning(
            "L-BFGS-B optimization did not converge: %s. "
            "Proceeding with best-found parameters — verify calibration manually.",
            result.message,
        )
    alpha, beta = float(result.x[0]), float(result.x[1])

    hybrids = [
        v + beta * (a - v) + alpha * nr
        for v, a, nr in zip(vina_scores, ad4_scores, n_residues_list)
    ]
    if len(hybrids) > 1:
        r, _ = pearsonr(hybrids, delta_g)
        pearson_r = float(r)
    else:
        pearson_r = float("nan")

    rmse = float(np.sqrt(np.mean([(h - d) ** 2 for h, d in zip(hybrids, delta_g)])))

    _log.info(
        "fit_calibration: alpha=%.4f beta=%.4f r=%.3f rmse=%.3f", alpha, beta, pearson_r, rmse
    )
    return {
        "alpha": alpha,
        "beta": beta,
        "pearson_r": pearson_r,
        "rmse_kcal_mol": rmse,
    }
