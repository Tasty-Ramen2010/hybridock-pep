"""Scoring package for HybriDock-Pep.

Exports the public scoring API: Vina, AD4, and entropy correction.
"""

from __future__ import annotations

from hybridock_pep.scoring.vina import check_grid_boundary, score_vina_batch
from hybridock_pep.scoring.ad4 import score_ad4_batch
from hybridock_pep.scoring.entropy import (
    load_calibration,
    write_calibration,
    apply_hybrid_score,
    fit_calibration,
)
from hybridock_pep.scoring.ensemble import (
    EnsembleCalibration,
    fit_ensemble_calibration,
    score as ensemble_score,
)

__all__ = [
    "check_grid_boundary",
    "score_vina_batch",
    "score_ad4_batch",
    "load_calibration",
    "write_calibration",
    "apply_hybrid_score",
    "fit_calibration",
    "EnsembleCalibration",
    "fit_ensemble_calibration",
    "ensemble_score",
]
