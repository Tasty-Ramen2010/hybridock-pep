from __future__ import annotations

from hybridock_pep.models import DockConfig, PoseFailure, PoseRecord, ScoredPose
from hybridock_pep.scoring.anchoring import (
    AnchorResult,
    Reference,
    anchored_affinity,
    sequence_identity,
)
from hybridock_pep.scoring.charge_complementarity import charge_complementarity_score
from hybridock_pep.scoring.double_difference import (
    DoubleDiffResult,
    double_difference_dg,
    double_difference_selectivity,
)

__all__ = [
    "DockConfig",
    "PoseRecord",
    "ScoredPose",
    "PoseFailure",
    # Axis 2 — same-receptor calibration (FEP-grade relative accuracy at docking cost)
    "Reference",
    "AnchorResult",
    "anchored_affinity",
    "sequence_identity",
    "DoubleDiffResult",
    "double_difference_dg",
    "double_difference_selectivity",
    # Axis 3 — within-target selectivity ranking
    "charge_complementarity_score",
]
