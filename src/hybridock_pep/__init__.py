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
from hybridock_pep.scoring.interaction_map import (
    IFP_FEATURE_ORDER,
    compute_ifp,
    score_crystal_complex,
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
    # Axis 1 — crystal-pose interaction-map scoring enhancement (+0.10 r on crystal poses)
    "compute_ifp",
    "score_crystal_complex",
    "IFP_FEATURE_ORDER",
]
