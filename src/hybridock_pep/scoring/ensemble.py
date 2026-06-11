"""Geometry+Vina ensemble ΔG scorer (SCORE-ENS).

Combines two partially-independent predictors of binding ΔG:
  * geometry  — linear model over pocket + interface descriptors (pose-derived SASA/contacts)
  * vina      — AutoDock Vina --score_only energy on the same pose

Rationale (docs/e19_pocket_baseline_breakthrough.md, E21): on crystal-65 the geometry model
(r=0.576) and Vina-fit (r=0.527) fail on the SAME complexes (residual corr 0.74 — amphipathic
helices) yet have enough independent error that a 50/50 z-blend reaches r=0.620, RMSE 1.68,
clearing the literature ballpark (~0.62) — without the thousands of training complexes those
ML tools require. ESM / helicity / MD-entropy terms were all tested and OVERFIT 65 points;
the ensemble is the only lever that helped.

The "good Vina parts" variant uses Vina's INTERMOLECULAR term only (vina_mode="inter"),
dropping the rotatable-bond/torsion penalty that drives Vina's size bias (Vina total
correlates with peptide length at -0.75). See `score()` and the calibration JSON.

Calibration (feature weights, z-norm stats, blend weight, vina_mode) is fit on a reference
panel by `fit_ensemble_calibration` and stored as JSON; production loads it read-only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Feature order is part of the calibration contract — do not reorder without refitting.
POCKET_FEATURES = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis"]
INTERFACE_FEATURES = ["bsa_hyd", "sasa_hb", "sasa_sb", "arom_cc", "hb_count"]
GEOMETRY_FEATURES = POCKET_FEATURES + INTERFACE_FEATURES


@dataclass(frozen=True)
class EnsembleCalibration:
    """Frozen calibration for the geometry+Vina ensemble.

    Attributes:
        feature_names: Geometry feature order the linear weights apply to.
        geo_intercept: Intercept of the standardized linear geometry model (kcal/mol).
        geo_weights: Per-feature weights on standardized features (kcal/mol).
        geo_mean / geo_std: Standardization stats for geometry features.
        geo_pred_mean / geo_pred_std: Z-norm stats of the geometry prediction.
        vina_mean / vina_std: Z-norm stats of the Vina score used.
        blend: Weight on geometry in the z-blend; Vina gets (1 - blend).
        vina_mode: "total" (full Vina energy) or "inter" (intermolecular term only).
        y_mean / y_std: ΔG distribution stats, to map the z-blend back to kcal/mol.
    """

    feature_names: list[str]
    geo_intercept: float
    geo_weights: list[float]
    geo_mean: list[float]
    geo_std: list[float]
    geo_pred_mean: float
    geo_pred_std: float
    vina_mean: float
    vina_std: float
    blend: float
    vina_mode: str
    y_mean: float
    y_std: float

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleCalibration":
        d = json.loads(Path(path).read_text())
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=2))


def _geometry_prediction(features: dict[str, float], cal: EnsembleCalibration) -> float:
    x = np.array([float(features.get(f, 0.0)) for f in cal.feature_names])
    z = (x - np.array(cal.geo_mean)) / (np.array(cal.geo_std) + 1e-9)
    return float(cal.geo_intercept + np.dot(cal.geo_weights, z))


def score(
    features: dict[str, float],
    vina_total: float,
    cal: EnsembleCalibration,
    *,
    vina_inter: float | None = None,
) -> float:
    """Ensemble ΔG estimate (kcal/mol; more negative = tighter).

    Args:
        features: Geometry descriptors for the pose (keys in GEOMETRY_FEATURES).
        vina_total: Vina --score_only total energy for the pose.
        cal: Loaded calibration.
        vina_inter: Vina intermolecular term; required when cal.vina_mode == "inter".

    Returns:
        Blended ΔG estimate in kcal/mol.

    Raises:
        ValueError: If vina_mode is "inter" but vina_inter was not supplied.
    """
    if cal.vina_mode == "inter":
        if vina_inter is None:
            raise ValueError("calibration uses vina_mode='inter' but vina_inter was not provided")
        vina_val = vina_inter
    else:
        vina_val = vina_total

    geo_pred = _geometry_prediction(features, cal)
    zg = (geo_pred - cal.geo_pred_mean) / (cal.geo_pred_std + 1e-9)
    zv = (vina_val - cal.vina_mean) / (cal.vina_std + 1e-9)
    # Vina energy and ΔG share sign (both more-negative = tighter), so blend directly.
    z_blend = cal.blend * zg + (1.0 - cal.blend) * zv
    return float(cal.y_mean + z_blend * cal.y_std)


def fit_ensemble_calibration(
    records: list[dict],
    *,
    blend: float = 0.5,
    vina_mode: str = "total",
    feature_names: list[str] | None = None,
) -> EnsembleCalibration:
    """Fit the ensemble calibration on a reference panel.

    Each record needs the geometry feature keys, ``y`` (experimental ΔG), and a Vina
    score under ``vina`` (total) and/or ``vina_inter`` (intermolecular).

    Args:
        records: Reference complexes with features + y + vina score(s).
        blend: Geometry weight in the z-blend (Vina gets 1 - blend).
        vina_mode: "total" or "inter".
        feature_names: Override the default geometry feature order.

    Returns:
        A fitted EnsembleCalibration.
    """
    feats = feature_names or GEOMETRY_FEATURES
    X = np.array([[float(r.get(f, 0.0)) for f in feats] for r in records])
    y = np.array([float(r["y"]) for r in records])
    vkey = "vina_inter" if vina_mode == "inter" else "vina"
    v = np.array([float(r[vkey]) for r in records])

    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    A = np.column_stack([np.ones(len(Z)), Z])
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    geo_pred = A @ w

    return EnsembleCalibration(
        feature_names=list(feats),
        geo_intercept=float(w[0]),
        geo_weights=[float(x) for x in w[1:]],
        geo_mean=[float(x) for x in mu],
        geo_std=[float(x) for x in sd],
        geo_pred_mean=float(geo_pred.mean()),
        geo_pred_std=float(geo_pred.std()),
        vina_mean=float(v.mean()),
        vina_std=float(v.std()),
        blend=float(blend),
        vina_mode=vina_mode,
        y_mean=float(y.mean()),
        y_std=float(y.std()),
    )
