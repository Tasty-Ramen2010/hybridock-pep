"""Vina score_only batch scorer (SCORE-01).

Implements per-pose grid boundary validation and single-instance Vina Python API
scoring. Returns (list[ScoredPose], list[PoseFailure]) — never raises on
per-pose failures.

Key design decisions (from RESEARCH.md / STATE.md):
- One Vina instance per batch; set_ligand_from_file() called per pose.
- compute_vina_maps() called ONCE before the pose loop (all 22 atom types).
- float(v.score()[0]) used throughout — never raw numpy array comparisons.
- Vina SWIG bindings have no documented thread safety; sequential scoring only.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from hybridock_pep.models import DockConfig, PoseFailure, ScoredPose

try:
    from vina import Vina
except ImportError:  # score-env not active (e.g. during unit tests with mocks)
    Vina = None  # type: ignore[assignment,misc]


logger = logging.getLogger(__name__)


def check_grid_boundary(
    pdbqt_path: Path,
    site_coords: tuple[float, float, float],
    box_size: float,
) -> bool:
    """Return True if any heavy ATOM/HETATM atom in the PDBQT falls outside the grid box.

    Hydrogen atoms (element H or HD in cols 76-78; or name starting with 'H'
    when the element column is absent) are excluded from the check.  babel adds
    polar H to PDBQT for Gasteiger charges; these can lie marginally outside the
    grid even when all heavy atoms are within bounds — falsely flagging otherwise
    good poses.  Vina and AD4 grid-based scoring are not materially affected by H
    positions, so only heavy-atom placement determines whether a pose is clipped.

    Parses fixed-column PDB/PDBQT coordinate fields (cols 30-38 x, 38-46 y,
    46-54 z). Boundary is inclusive: an atom exactly on the edge is NOT clipped.
    Malformed coordinate lines are silently skipped.

    Args:
        pdbqt_path: Path to the prepared PDBQT file for one pose.
        site_coords: (cx, cy, cz) grid box center in Angstrom.
        box_size: Grid box edge length in Angstrom.

    Returns:
        True if any heavy atom lies strictly outside site_coords ± box_size/2 on
        any axis; False otherwise (including the case of no parseable atoms).
    """
    cx, cy, cz = site_coords
    half = box_size / 2.0

    for line in pdbqt_path.read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue

        # Skip hydrogens — babel-added H can lie marginally outside the grid
        # without affecting scoring accuracy (Fix I, 2026-04-30).
        element = line[76:78].strip() if len(line) > 76 else ""
        name = line[12:16].strip() if len(line) > 16 else ""
        if element in ("H", "HD") or (not element and name.startswith("H")):
            continue

        if (
            x < cx - half
            or x > cx + half
            or y < cy - half
            or y > cy + half
            or z < cz - half
            or z > cz + half
        ):
            return True

    return False


def _append_clipped_pose(path: Path, pose_idx: int, pdbqt_path: Path | None) -> None:
    """Append a clipped-pose entry to the run_metadata.json file.

    Reads an existing JSON file if present, appends to the "clipped_poses"
    list, and writes back atomically. Malformed JSON is silently overwritten.
    Parent directories are created if they do not exist.

    Args:
        path: Absolute path to the metadata JSON file.
        pose_idx: Index of the clipped pose.
        pdbqt_path: Path to the clipped pose's PDBQT file (may be None).
    """
    if not path.exists():
        data: dict = {}
    else:
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            data = {}

    data.setdefault("clipped_poses", []).append(
        {"pose_idx": pose_idx, "pdbqt_path": str(pdbqt_path)}
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)  # atomic on POSIX; overwrites destination


def score_vina_batch(
    poses: list[ScoredPose],
    config: DockConfig,
    receptor_pdbqt: Path,
    *,
    verbosity: int = 0,
    metadata_path: Path | None = None,
) -> tuple[list[ScoredPose], list[PoseFailure]]:
    """Score a batch of poses with Vina --score_only using a single Vina instance.

    Creates one Vina instance, loads the receptor once, computes maps once
    before the pose loop, then calls set_ligand_from_file() per pose.
    Per-pose exceptions are caught and recorded as PoseFailure; the batch
    never aborts on a single bad pose.

    Clipped poses (atoms outside grid bounds) are flagged with is_clipped=True,
    a WARNING is logged, and the pose entry is appended to run_metadata.json
    (SCORE-01).

    Note: Vina SWIG bindings are not thread-safe; poses are scored sequentially.

    Args:
        poses: List of ScoredPose objects; each must have pdbqt_path set.
        config: Validated DockConfig supplying site_coords and box_size.
        receptor_pdbqt: Path to the prepared receptor PDBQT file.
        verbosity: Vina verbosity level (0=silent). Default 0.
        metadata_path: If provided, clipped pose entries are appended to this
            JSON file. Parent directories are created if absent.

    Returns:
        A tuple (scored, failures) where scored contains successfully scored
        ScoredPose objects and failures contains PoseFailure records for poses
        that raised an exception.

    Raises:
        Exception: Exceptions during Vina instance creation or receptor loading
            propagate to the caller (not silently swallowed); only per-pose
            scoring exceptions are caught.
    """
    scored: list[ScoredPose] = []
    failures: list[PoseFailure] = []

    # --- One instance; receptor loaded once; maps computed once before loop ---
    v = Vina(sf_name="vina", verbosity=verbosity)
    v.set_receptor(str(receptor_pdbqt))
    v.compute_vina_maps(
        center=list(config.site_coords),
        box_size=[config.box_size] * 3,
    )

    logger.info("Vina scorer: %d poses, receptor=%s", len(poses), receptor_pdbqt)

    for pose in poses:
        try:
            if pose.pdbqt_path is None:
                raise ValueError(
                    f"Pose {pose.pose_idx} has pdbqt_path=None; "
                    "was prep/ligand.py run before scoring?"
                )
            pose.is_clipped = check_grid_boundary(
                pose.pdbqt_path, config.site_coords, config.box_size
            )
            if pose.is_clipped:
                logger.warning(
                    "Pose %d: atoms outside grid bounds (is_clipped=True)", pose.pose_idx
                )
                if metadata_path is not None:
                    _append_clipped_pose(metadata_path, pose.pose_idx, pose.pdbqt_path)

            v.set_ligand_from_file(str(pose.pdbqt_path))
            pose.vina_score = float(v.score()[0])
            scored.append(pose)

        except Exception as e:  # noqa: BLE001 — per-pose isolation required
            logger.warning("Pose %d scoring failed: %s: %s", pose.pose_idx, type(e).__name__, e)
            failures.append(
                PoseFailure(
                    pose_idx=pose.pose_idx,
                    stage="scoring",
                    error_msg=f"{type(e).__name__}: {e}",
                )
            )

    return scored, failures
