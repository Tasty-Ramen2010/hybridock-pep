"""Pose PDB parser — converts pose_*.pdb files into PoseRecord objects (SAMP-01).

Implements collect-all-failures batch semantics: all files are processed regardless
of individual failures. Malformed PDB files produce PoseFailure(stage="parsing")
records; the batch never raises on per-pose errors (D-12).

Cα coordinates are extracted at parse time and stored in PoseRecord.ca_coords
(shape [n_residues, 3], dtype float64) so downstream clustering can access them
in O(1) without re-reading disk (D-13).

Sequence is extracted per D-14 (locked decision): SEQRES records first, falling
back to ATOM record residue names when SEQRES is absent.  MDAnalysis-written PDB
files (RAPiDock output) typically lack SEQRES records (Pitfall 5 in RESEARCH.md),
so the ATOM fallback is the common production path.  SEQRES-first is still required
for correctness when reference PDB files are passed via --input-poses.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from hybridock_pep.models import PoseFailure, PoseRecord

logger = logging.getLogger(__name__)

try:
    from Bio.Data.IUPACData import protein_letters_3to1 as _prot_3to1

    _STANDARD_AA: frozenset[str] = frozenset(k.upper() for k in _prot_3to1)
except ImportError:
    _STANDARD_AA = frozenset()


def _is_standard_aa(residue: object) -> bool:
    """Return True if residue is one of the 20 standard amino acids.

    Replaces deprecated Bio.PDB.Polypeptide.is_aa (deprecated ≥1.80).
    """
    try:
        return residue.get_resname().strip().upper() in _STANDARD_AA  # type: ignore[union-attr]
    except AttributeError:
        return False


def parse_poses(
    poses_dir: Path,
) -> tuple[list[PoseRecord], list[PoseFailure]]:
    """Parse all pose_*.pdb files in poses_dir into PoseRecord objects.

    All poses are processed regardless of individual failures.  Per-pose
    exceptions are caught and recorded as PoseFailure records (D-12).
    The batch never raises.

    Args:
        poses_dir: Directory containing pose_0.pdb ... pose_N.pdb files as
                   written by rapidock_runner.py after rank→pose renaming.

    Returns:
        Tuple of (records, failures):
            records: Successfully parsed PoseRecord objects.  Each has ca_coords
                     populated (shape [n_residues, 3], float64).
            failures: PoseFailure records for any file that could not be parsed.
    """
    records: list[PoseRecord] = []
    failures: list[PoseFailure] = []

    pdb_files = sorted(poses_dir.glob("pose_*.pdb"))
    logger.info("Parsing %d pose PDB files from %s", len(pdb_files), poses_dir)

    for pdb_path in pdb_files:
        try:
            pose_idx = int(pdb_path.stem.split("_")[1])
        except (ValueError, IndexError) as e:
            # Should not happen after renaming, but be defensive
            failures.append(
                PoseFailure(
                    pose_idx=-1,
                    stage="parsing",
                    error_msg=f"Unrecognised filename {pdb_path.name}: {e}",
                )
            )
            logger.warning("Skipping %s: unrecognised filename pattern", pdb_path.name)
            continue

        try:
            record = _parse_single_pose(pose_idx, pdb_path)
            records.append(record)
        except Exception as e:  # noqa: BLE001 — PDBParser raises varied exceptions
            failures.append(
                PoseFailure(
                    pose_idx=pose_idx,
                    stage="parsing",
                    error_msg=f"{type(e).__name__}: {e}",
                )
            )
            logger.warning("Pose %d parse failed: %s", pose_idx, e)

    logger.info(
        "Pose parsing complete: %d succeeded, %d failed",
        len(records),
        len(failures),
    )
    return records, failures


def _parse_single_pose(pose_idx: int, pdb_path: Path) -> PoseRecord:
    """Parse one PDB file into a PoseRecord with Cα coordinates.

    Uses Biopython PDBParser (QUIET=True) to suppress REMARK/SSBOND warnings
    that RAPiDock PDB files frequently trigger.

    Sequence extraction follows D-14 (locked decision):
    1. Try SEQRES records first (lines beginning with "SEQRES" in the raw file).
    2. If SEQRES present and parseable → use as sequence.
    3. If SEQRES absent or unparseable → fall back to ATOM record residue names
       via three_to_one().
    4. If neither yields any residues → raise ValueError → caller emits PoseFailure.

    Note: MDAnalysis-written PDB files (RAPiDock output) typically lack SEQRES
    records (RESEARCH.md Pitfall 5), so the ATOM fallback is the common
    production path.

    Args:
        pose_idx: Zero-based index of this pose.
        pdb_path: Path to the PDB file.

    Returns:
        PoseRecord with pose_idx, pdb_path (absolute), sequence, ca_coords.

    Raises:
        ValueError: If no standard amino-acid CA atoms found (empty peptide).
        Exception: Any PDBParser internal error propagates up to parse_poses().
    """
    from Bio.Data.IUPACData import protein_letters_3to1  # three_to_one removed in Biopython 1.80+
    from Bio.PDB import PDBParser  # local import — biopython optional dep

    _three_to_one = {k.upper(): v for k, v in protein_letters_3to1.items()}

    def three_to_one(resname: str) -> str:
        result = _three_to_one.get(resname.upper())
        if result is None:
            raise KeyError(resname)
        return result

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(f"pose_{pose_idx}", str(pdb_path))

    ca_coords_list: list[list[float]] = []

    # Iterate over the first MODEL only (RAPiDock writes single-model PDBs)
    model = next(iter(structure))
    for chain in model:
        for residue in chain:
            if not _is_standard_aa(residue):
                continue
            if "CA" not in residue:
                continue
            ca_coords_list.append(list(residue["CA"].get_vector().get_array()))

    if not ca_coords_list:
        raise ValueError(f"No standard amino-acid CA atoms found in {pdb_path}")

    ca_coords = np.array(ca_coords_list, dtype=np.float64)  # shape [n_res, 3]

    # D-14: SEQRES-first sequence extraction with ATOM fallback
    sequence = _extract_sequence_seqres_first(pdb_path, model, three_to_one)

    return PoseRecord(
        pose_idx=pose_idx,
        pdb_path=pdb_path.resolve(),
        sequence=sequence,
        ca_coords=ca_coords,
    )


def _extract_sequence_seqres_first(
    pdb_path: Path,
    model: object,
    three_to_one: object,
) -> str:
    """Extract peptide sequence per D-14: SEQRES first, ATOM fallback.

    Args:
        pdb_path: Path to PDB file (for raw SEQRES line parsing).
        model: Biopython Model object (first MODEL in structure).
        three_to_one: Callable mapping 3-letter residue name to 1-letter code.

    Returns:
        Single-letter amino acid sequence string.

    Raises:
        ValueError: If neither SEQRES nor ATOM records yield any residues.
    """
    # --- Step 1: Try SEQRES records (D-14 primary path) ---
    # Only read SEQRES for chains that actually have standard-AA CA atoms in ATOM records.
    # For co-crystal PDBs passed via --input-poses, SEQRES includes receptor chains too —
    # filtering prevents concatenating receptor+peptide residues into the sequence field.
    atom_chains: set[str] = {
        chain.id
        for chain in model  # type: ignore[union-attr]
        for residue in chain
        if _is_standard_aa(residue) and "CA" in residue
    }
    seqres_residues: list[str] = []
    try:
        raw_lines = pdb_path.read_text(errors="replace").splitlines()
        for line in raw_lines:
            if not line.startswith("SEQRES"):
                continue
            # SEQRES col 12 (0-indexed 11) is the chain ID
            chain_id = line[11] if len(line) > 11 else " "
            if chain_id not in atom_chains:
                continue
            # cols 19+ contain space-separated residue names
            residue_names = line[19:].split()
            for resname in residue_names:
                try:
                    seqres_residues.append(three_to_one(resname))
                except KeyError:
                    seqres_residues.append("X")
    except OSError:
        seqres_residues = []

    if seqres_residues:
        logger.debug("Sequence from SEQRES records: %d residues", len(seqres_residues))
        return "".join(seqres_residues)

    # --- Step 2: Fall back to ATOM record residue names (D-14 fallback) ---
    # Common for MDAnalysis-written PDB files (RESEARCH.md Pitfall 5)
    atom_residues: list[str] = []
    for chain in model:
        for residue in chain:
            if not _is_standard_aa(residue):
                continue
            try:
                atom_residues.append(three_to_one(residue.get_resname()))
            except KeyError:
                atom_residues.append("X")

    if atom_residues:
        logger.debug("Sequence from ATOM records (SEQRES absent): %d residues", len(atom_residues))
        return "".join(atom_residues)

    # --- Step 3: Neither parseable → PoseFailure will be emitted by caller ---
    raise ValueError(
        f"No sequence could be extracted from {pdb_path} (no SEQRES or ATOM residues)"
    )
