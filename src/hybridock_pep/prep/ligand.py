from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from hybridock_pep.models import PoseFailure

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Module-level worker (must NOT be a closure — required for ProcessPoolExecutor
# pickling on POSIX systems using the 'spawn' start method on macOS).
# --------------------------------------------------------------------------- #


def _prepare_single_ligand(
    args: tuple[int, Path, Path],
) -> Path | PoseFailure:
    """Convert one pose PDB to PDBQT using Meeko.

    This function is intentionally at module level to satisfy ProcessPoolExecutor
    pickling requirements. Do not move it inside prepare_ligand_batch.

    Meeko assigns Gasteiger partial charges automatically during from_pdb().
    AD4 scoring consumes these charges explicitly; Vina ignores them (§2.1).

    Args:
        args: Tuple of (pose_idx, pdb_path, output_dir).

    Returns:
        Path to the written PDBQT on success, or PoseFailure on any error.
    """
    pose_idx, pdb_path, output_dir = args
    pdbqt_path = Path(output_dir) / (Path(pdb_path).stem + ".pdbqt")

    try:
        from meeko import MoleculePreparation  # local import — avoids top-level meeko dep in main process
        mols = MoleculePreparation.from_pdb(str(pdb_path))
        if not mols:
            return PoseFailure(
                pose_idx=pose_idx,
                stage="prep",
                error_msg=f"Meeko returned no molecules for {pdb_path}",
            )
        pdbqt_string = mols[0].write_pdbqt_string()
        Path(pdbqt_path).write_text(pdbqt_string)
        return Path(pdbqt_path)
    except Exception as e:  # noqa: BLE001
        # Meeko raises varied internal errors (ValueError, KeyError, AttributeError).
        # Collect all as PoseFailure — never propagate from the worker.
        return PoseFailure(
            pose_idx=pose_idx,
            stage="prep",
            error_msg=f"{type(e).__name__}: {e}",
        )


def prepare_ligand_batch(
    pdb_paths: list[Path],
    output_dir: Path,
    *,
    max_workers: int | None = None,
) -> tuple[list[Path], list[PoseFailure]]:
    """Convert a list of pose PDB files to PDBQT in parallel using Meeko.

    All poses are processed regardless of individual failures. Failures are
    collected into PoseFailure records and returned alongside successes. The
    caller decides how many failures are acceptable — this function never
    raises on per-pose errors.

    Gasteiger charges are assigned automatically by Meeko and are required
    for AD4 scoring (§2.1 — Vina ignores them, AD4 uses them explicitly).

    Args:
        pdb_paths: List of pose PDB paths to convert.
        output_dir: Directory to write PDBQT files into. Created if absent.
        max_workers: Number of worker processes. None → os.cpu_count().

    Returns:
        Tuple of (pdbqt_paths, failures) where:
        - pdbqt_paths: Paths of successfully written PDBQT files.
        - failures: PoseFailure records for any pose that could not be converted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_workers = max_workers or os.cpu_count()

    args_list = [
        (idx, pdb_path, output_dir)
        for idx, pdb_path in enumerate(pdb_paths)
    ]

    successes: list[Path] = []
    failures: list[PoseFailure] = []

    logger.info(
        "Preparing %d pose PDBs → PDBQT (workers=%s)", len(pdb_paths), effective_workers
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_prepare_single_ligand, args): args[0]
            for args in args_list
        }
        for future in as_completed(future_to_idx):
            result = future.result()
            if isinstance(result, PoseFailure):
                failures.append(result)
                logger.warning("Pose %d prep failed: %s", result.pose_idx, result.error_msg)
            else:
                successes.append(result)

    logger.info(
        "Ligand prep complete: %d succeeded, %d failed", len(successes), len(failures)
    )
    return successes, failures
