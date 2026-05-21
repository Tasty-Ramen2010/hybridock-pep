from __future__ import annotations

import logging
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from hybridock_pep.models import PoseFailure

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Module-level worker (must NOT be a closure — required for ProcessPoolExecutor
# pickling on POSIX systems using the 'spawn' start method on macOS).
# --------------------------------------------------------------------------- #


def _wrap_pdbqt_rigid(flat_pdbqt_path: Path) -> None:
    """Wrap a flat (no torsion tree) PDBQT in ROOT/ENDROOT/TORSDOF 0.

    babel -xr generates a flat PDBQT where all atoms are listed without
    ROOT/ENDROOT/BRANCH/ENDBRANCH tags. Vina 1.2.x Python API requires exactly
    one ROOT/ENDROOT block per ligand. This function post-processes the file
    in-place: REMARK lines are preserved, all ATOM/HETATM lines are wrapped in
    a single ROOT/ENDROOT block, and TORSDOF 0 is appended.

    For peptide poses we always use TORSDOF 0 (rigid ligand for --score_only)
    because the pose is already fully specified — no flexible docking is done.

    Args:
        flat_pdbqt_path: Path to the flat PDBQT file to modify in-place.
    """
    lines = flat_pdbqt_path.read_text().splitlines(keepends=True)
    remarks = [l for l in lines if l.startswith("REMARK")]
    atoms = [l for l in lines if l.startswith("ATOM") or l.startswith("HETATM")]
    wrapped = "".join(remarks) + "ROOT\n" + "".join(atoms) + "ENDROOT\nTORSDOF 0\n"
    flat_pdbqt_path.write_text(wrapped)


def _prepare_single_ligand(
    args: tuple[int, Path, Path],
) -> Path | PoseFailure:
    """Convert one pose PDB to PDBQT.

    Routes phospho-residue poses (TPO/SEP/PTR) through Meeko's Polymer API,
    which natively handles phosphate group atom types and Gasteiger charges.
    Standard peptides continue through the babel path (unchanged behaviour).

    This function is intentionally at module level to satisfy ProcessPoolExecutor
    pickling requirements. Do not move it inside prepare_ligand_batch.

    Uses babel -xr (rigid mode) which outputs a flat PDBQT without a torsion
    tree. The flat output is then wrapped in ROOT/ENDROOT/TORSDOF 0 by
    _wrap_pdbqt_rigid so that Vina 1.2.x Python API accepts it.

    babel -h without -xr fragments peptide PDB into multiple ROOT/ENDROOT
    molecules (one per connectivity component since RAPiDock PDB output lacks
    CONECT records), and Vina 1.2.x rejects multi-ROOT PDBQT with "Unknown or
    inappropriate tag found in flex residue or ligand. > ROOT".

    Gasteiger partial charges are still assigned by babel and present in the
    PDBQT (required for AD4 scoring; Vina ignores them per §2.1).

    Args:
        args: Tuple of (pose_idx, pdb_path, output_dir).

    Returns:
        Path to the written PDBQT on success, or PoseFailure on any error.
    """
    pose_idx, pdb_path, output_dir = args

    # Phospho-residue fast path — Meeko handles TPO/SEP/PTR natively.
    from hybridock_pep.prep.phospho import has_phospho_residues, prepare_phospho_ligand
    if has_phospho_residues(Path(pdb_path)):
        logger.debug("Pose %d has phospho residues — routing through Meeko", pose_idx)
        return prepare_phospho_ligand(pose_idx, Path(pdb_path), Path(output_dir))

    pdbqt_path = Path(output_dir) / (Path(pdb_path).stem + ".pdbqt")

    try:
        babel_bin = shutil.which("babel")
        if babel_bin is None:
            raise FileNotFoundError(
                "babel not found on PATH — install ADFRsuite and add its bin/ to PATH"
            )
        result = subprocess.run(
            [babel_bin, "-i", "pdb", str(pdb_path), "-o", "pdbqt", str(pdbqt_path), "-h", "-xr"],
            capture_output=True,
            text=True,
        )
        # babel exits 0 and creates an empty file on input errors — check both
        if result.returncode != 0 or not pdbqt_path.exists() or pdbqt_path.stat().st_size == 0:
            raise RuntimeError(
                f"babel exited {result.returncode} with empty/missing output: "
                f"{result.stderr.strip()}"
            )
        _wrap_pdbqt_rigid(pdbqt_path)
        return pdbqt_path
    except Exception as e:  # noqa: BLE001
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
    """Convert a list of pose PDB files to PDBQT in parallel using babel.

    All poses are processed regardless of individual failures. Failures are
    collected into PoseFailure records and returned alongside successes. The
    caller decides how many failures are acceptable — this function never
    raises on per-pose errors.

    Gasteiger charges are assigned by babel (-h flag) and are required for AD4
    scoring (§2.1 — Vina ignores them, AD4 uses them explicitly).

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

    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
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
