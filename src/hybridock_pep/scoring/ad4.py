"""AD4 batch scorer (SCORE-02).

Implements per-pose AD4 scoring via Vina(sf_name='ad4') with load_maps().
CRITICAL: The receptor-loading API for AD4 mode is load_maps(map_prefix),
NOT the receptor-setter method used by the Vina scorer. Calling the receptor
setter with sf_name='ad4' raises RuntimeError from the C++ binding.

Key design decisions:
- One Vina(sf_name='ad4') instance per batch; load_maps() called ONCE before loop.
- Per-pose: set_ligand_from_file() → float(v.score()[0]).
- is_ad4_anomaly=True when ad4_score > 0 (repulsive / unphysical); pose still
  included in scored list — informational flag per D-06.
- Defensive HD map existence check before load_maps() for clear diagnostics.
- Per-pose exception → PoseFailure(stage="scoring"); batch never aborts (D-07).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from hybridock_pep.models import PoseFailure, ScoredPose

if TYPE_CHECKING:
    from vina import Vina as _VinaType

try:
    from vina import Vina
except ImportError:  # score-env not active (e.g. during unit tests with mocks)
    Vina = None  # type: ignore[assignment,misc]


logger = logging.getLogger(__name__)


def score_ad4_batch(
    poses: list[ScoredPose],
    maps_dir: Path,
    *,
    verbosity: int = 0,
) -> tuple[list[ScoredPose], list[PoseFailure]]:
    """Score a batch of poses with AutoDock4 using a single Vina(sf_name='ad4') instance.

    Creates one Vina instance with sf_name='ad4', loads pre-computed AD4 grid
    maps once via load_maps() before the pose loop, then calls
    set_ligand_from_file() per pose. Per-pose exceptions are caught and recorded
    as PoseFailure; the batch never aborts on a single bad pose (D-07).

    Poses with ad4_score > 0 (repulsive/unphysical) are flagged with
    is_ad4_anomaly=True but are still included in the scored list (D-06).

    Note: The receptor-setter API must NOT be used when sf_name='ad4' is
    active — the Vina C++ binding raises RuntimeError. Use load_maps() only.

    Args:
        poses: List of ScoredPose objects; each must have pdbqt_path set.
        maps_dir: Directory containing pre-computed AD4 map files produced by
            prep/grids.py. Expected files: receptor.HD.map, receptor.C.map, etc.
            The map prefix passed to load_maps() is str(maps_dir / "receptor").
        verbosity: Vina verbosity level (0=silent). Default 0.

    Returns:
        A tuple (scored, failures) where scored contains successfully scored
        ScoredPose objects (with ad4_score and is_ad4_anomaly set) and failures
        contains PoseFailure records for poses that raised an exception.

    Raises:
        FileNotFoundError: If the required receptor.HD.map is absent from
            maps_dir. Run prep/grids.py first to generate AD4 grid maps.
        Exception: Exceptions during Vina instance creation or load_maps() call
            propagate to the caller; only per-pose scoring exceptions are caught.
    """
    # Belt-and-suspenders: verify HD map exists before calling load_maps()
    hd_map = maps_dir / "receptor.HD.map"
    if not hd_map.exists():
        raise FileNotFoundError(
            f"AD4 HD map not found: {hd_map}. Run prep/grids.py first."
        )

    map_prefix = str(maps_dir / "receptor")
    logger.info("AD4 scorer: %d poses, maps_prefix=%s", len(poses), map_prefix)

    # One instance; maps loaded once — do NOT use the receptor-setter with sf_name='ad4'
    v = Vina(sf_name="ad4", verbosity=verbosity)
    v.load_maps(map_prefix)

    scored: list[ScoredPose] = []
    failures: list[PoseFailure] = []

    for pose in poses:
        try:
            v.set_ligand_from_file(str(pose.pdbqt_path))
            pose.ad4_score = float(v.score()[0])
            pose.is_ad4_anomaly = pose.ad4_score > 0
            if pose.is_ad4_anomaly:
                logger.warning(
                    "Pose %d: AD4 anomaly (positive score=%.3f kcal/mol)",
                    pose.pose_idx,
                    pose.ad4_score,
                )
            scored.append(pose)

        except Exception as e:  # noqa: BLE001 — per-pose isolation required (D-07)
            logger.warning(
                "Pose %d AD4 scoring failed: %s: %s", pose.pose_idx, type(e).__name__, e
            )
            failures.append(
                PoseFailure(
                    pose_idx=pose.pose_idx,
                    stage="scoring",
                    error_msg=f"{type(e).__name__}: {e}",
                )
            )

    return scored, failures
