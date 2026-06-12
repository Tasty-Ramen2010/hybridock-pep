"""Length-conditional scoring router.

Short peptides (<= ``SHORT_MAX_LEN`` residues) are a distinct binding regime: with few interface contacts,
their affinity is dominated by hydrophobic burial, and the 16-feature geometry model fits them with the
wrong coefficients (13 of its features have near-zero dynamic range on short peptides, injecting noise).
On the pooled crystal-65 + the-98 benchmark this collapsed short-peptide ranking to r~0 (slope 0.03).

Routing short peptides to a lean 3-feature hydrophobic sub-model recovers them:
  short-bin LOO  r 0.02 -> 0.51 ,  RMSE 1.79 -> 1.21 kcal/mol  (n=22)
  held-out test  pooled r 0.603 -> 0.682 , RMSE 1.77 -> 1.61   (rest of the set unchanged)
See docs/e19_pocket_baseline_breakthrough.md (E85-E87) for the full length-stratified analysis.

The sub-model uses only features already in ``ensemble.GEOMETRY_FEATURES`` (no new extraction). Long and
very-long peptides did NOT benefit from their own sub-models (too few samples; signal is conformational,
needs MD) and are left on the standard ensemble path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ensemble import EnsembleCalibration, score as ensemble_score

# Threshold and lean feature set are part of the calibration contract — refit if changed.
SHORT_MAX_LEN = 8
SHORT_FEATURES = ["bsa_hyd", "mj_contact", "strength_bur"]


@dataclass(frozen=True)
class LengthRouterCalibration:
    """Frozen short-peptide sub-model (standardized linear ridge on SHORT_FEATURES).

    Attributes:
        short_max_len: Peptides with length <= this route to the short sub-model.
        feature_names: Feature order the short weights apply to (subset of GEOMETRY_FEATURES).
        intercept: Intercept in kcal/mol.
        weights: Per-feature weights on standardized features (kcal/mol).
        mean / std: Standardization stats for the short features.
    """

    short_max_len: int
    feature_names: list[str]
    intercept: float
    weights: list[float]
    mean: list[float]
    std: list[float]

    @classmethod
    def load(cls, path: str | Path) -> "LengthRouterCalibration":
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=2))


def _short_prediction(features: dict[str, float], cal: LengthRouterCalibration) -> float:
    x = np.array([float(features.get(f, 0.0)) for f in cal.feature_names])
    z = (x - np.array(cal.mean)) / (np.array(cal.std) + 1e-9)
    return float(cal.intercept + np.dot(cal.weights, z))


def route_score(
    features: dict[str, float],
    vina_total: float,
    peptide_length: int,
    router_cal: LengthRouterCalibration,
    ensemble_cal: EnsembleCalibration,
    *,
    vina_inter: float | None = None,
) -> float:
    """Length-routed ΔG estimate (kcal/mol; more negative = tighter).

    Short peptides use the lean hydrophobic sub-model directly (Vina is noisy on few-contact poses);
    all others use the standard geometry+Vina ensemble unchanged.

    Args:
        features: Geometry descriptors for the pose (keys in ensemble.GEOMETRY_FEATURES).
        vina_total: Vina --score_only total energy for the pose.
        peptide_length: Number of residues in the peptide (routing key).
        router_cal: Short-peptide sub-model calibration.
        ensemble_cal: Standard ensemble calibration (used for non-short peptides).
        vina_inter: Vina intermolecular term; forwarded to the ensemble when it uses vina_mode='inter'.

    Returns:
        ΔG estimate in kcal/mol.
    """
    if peptide_length <= router_cal.short_max_len:
        return _short_prediction(features, router_cal)
    return ensemble_score(features, vina_total, ensemble_cal, vina_inter=vina_inter)


def fit_length_router(
    records: list[dict],
    *,
    short_max_len: int = SHORT_MAX_LEN,
    feature_names: list[str] | None = None,
    ridge: float = 1.0,
) -> LengthRouterCalibration:
    """Fit the short-peptide sub-model on the short members of a reference panel.

    Args:
        records: Reference panel; each needs SHORT_FEATURES, ``y`` (experimental ΔG), and ``length``.
        short_max_len: Length threshold defining the short regime.
        feature_names: Override the short feature order (default SHORT_FEATURES).
        ridge: L2 strength on standardized weights.

    Returns:
        A frozen LengthRouterCalibration.

    Raises:
        ValueError: If fewer than 8 short-peptide records are available to fit.
    """
    cols = feature_names or SHORT_FEATURES
    short = [r for r in records if int(r["length"]) <= short_max_len]
    X = np.array([[float(r[c]) for c in cols] for r in short], float)
    y = np.array([float(r["y"]) for r in short], float)
    ok = ~np.isnan(X).any(axis=1)
    X, y = X[ok], y[ok]
    if len(X) < 8:
        raise ValueError(f"need >=8 short-peptide records to fit, got {len(X)}")
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    A = np.column_stack([np.ones(len(Z)), Z])
    R = np.eye(A.shape[1]) * ridge
    R[0, 0] = 0.0
    w = np.linalg.solve(A.T @ A + R, A.T @ y)
    return LengthRouterCalibration(
        short_max_len=short_max_len,
        feature_names=list(cols),
        intercept=float(w[0]),
        weights=[float(v) for v in w[1:]],
        mean=[float(v) for v in mu],
        std=[float(v) for v in sd],
    )
