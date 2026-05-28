from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from hybridock_pep.models import PoseFailure

logger = logging.getLogger(__name__)

# Fraction of ATOM/HETATM records that must be named UNL before we consider
# the PDBQT malformed (babel failed to parse amino-acid residue names).
# Crystal-pose PDBs with ionisation-state markers (N1+, O1-) in the element
# column can trigger this babel failure → all atoms get labelled UNL →
# wrong Gasteiger charges for every atom in the peptide.
_UNL_FRACTION_THRESHOLD = 0.10  # >10 % UNL atoms → warn


def sanitize_pdb_for_babel(pdb_path: Path, out_path: Path) -> None:
    """Write a babel-safe version of a crystal-pose PDB to out_path.

    Crystal structures processed by AMBER/CHARMM prep tools embed ionisation-
    state markers in the PDB element column (cols 77-78): ``N1+`` for protonated
    nitrogen, ``O1-`` for deprotonated oxygen.  babel uses that column to
    determine the element; seeing ``N1+`` it cannot match any known element and
    falls back to treating the whole molecule as an unknown ligand (``UNL``),
    which corrupts the Gasteiger charge assignment for every atom.

    This function strips the ionisation markers in-place on a copy, leaving the
    element column as a plain 1-2 character symbol (``N``, ``O``, ``S``, …).

    Args:
        pdb_path: Source PDB (read-only).
        out_path: Destination for the cleaned PDB (written by this function).
    """
    _ELEM_FIX = re.compile(r"([CNOS])(\d[+-]|[+-])")
    lines = pdb_path.read_text().splitlines(keepends=True)
    cleaned: list[str] = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 76:
            # cols 76-80 (0-indexed): element symbol, may have charge annotation
            suffix = line[76:].rstrip("\n")
            suffix_clean = _ELEM_FIX.sub(r"\1 ", suffix).rstrip() + "\n"
            line = line[:76] + suffix_clean
        cleaned.append(line)
    out_path.write_text("".join(cleaned))


def _count_unl_fraction(pdbqt_path: Path) -> float:
    """Return the fraction of ATOM/HETATM records labelled residue ``UNL``.

    Args:
        pdbqt_path: PDBQT file to inspect.

    Returns:
        Float in [0.0, 1.0].  Returns 0.0 if the file has no ATOM/HETATM lines.
    """
    total = 0
    unl = 0
    for line in pdbqt_path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            total += 1
            if len(line) >= 20 and line[17:20].strip() == "UNL":
                unl += 1
    return (unl / total) if total else 0.0


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


def _try_prepare_ligand4_fallback(pdb_path: Path, pdbqt_out: Path) -> Path | None:
    """Attempt PDBQT prep via ADFRsuite prepare_ligand4.py when babel produces UNL.

    prepare_ligand4.py (AutoDockTools-based) correctly identifies amino-acid
    residues and assigns per-residue Gasteiger charges for crystal-pose PDBs
    that confuse babel (unusual atom names, N1+/O1- element markers, etc.).

    The output is a flexible PDBQT with a full torsion tree.  This function
    strips the BRANCH/ENDBRANCH torsion tree and rewraps with ROOT/ENDROOT/
    TORSDOF 0 (rigid ligand) so it is compatible with Vina score_only mode.

    Args:
        pdb_path: Source PDB path.
        pdbqt_out: Destination PDBQT path (overwritten on success).

    Returns:
        pdbqt_out on success, None if prepare_ligand4.py is not available or
        fails.
    """
    # Locate ADFRsuite pythonsh and prepare_ligand4.py
    pythonsh = shutil.which("pythonsh")
    if pythonsh is None:
        # Try relative to prepare_receptor (same ADFRsuite bin/)
        prep_rec = shutil.which("prepare_receptor")
        if prep_rec:
            pythonsh = str(Path(prep_rec).parent / "pythonsh")
        if not pythonsh or not Path(pythonsh).exists():
            return None

    # Locate prepare_ligand4.py — typically at
    # {ADFR_ROOT}/CCSBpckgs/AutoDockTools/Utilities24/prepare_ligand4.py
    prep4 = shutil.which("prepare_ligand4.py")
    if prep4 is None:
        adfr_root = Path(pythonsh).resolve().parent.parent
        candidate = (
            adfr_root / "CCSBpckgs" / "AutoDockTools" / "Utilities24" / "prepare_ligand4.py"
        )
        prep4 = str(candidate) if candidate.exists() else None
    if prep4 is None:
        return None

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as tmp:
            tmp_out = Path(tmp.name)

        result = subprocess.run(
            [pythonsh, prep4, "-l", str(pdb_path.resolve()), "-o", str(tmp_out)],
            capture_output=True,
            text=True,
            cwd=str(pdb_path.parent),  # prepare_ligand4.py resolves relative paths from cwd
        )
        if result.returncode != 0 or not tmp_out.exists() or tmp_out.stat().st_size == 0:
            tmp_out.unlink(missing_ok=True)
            return None

        # Strip torsion tree → make rigid for score_only
        lines = tmp_out.read_text().splitlines(keepends=True)
        remarks = [l for l in lines if l.startswith("REMARK")]
        atoms = [
            l for l in lines
            if l.startswith("ATOM") or l.startswith("HETATM")
        ]
        pdbqt_out.write_text(
            "".join(remarks) + "ROOT\n" + "".join(atoms) + "ENDROOT\nTORSDOF 0\n"
        )
        tmp_out.unlink(missing_ok=True)
        return pdbqt_out

    except Exception:  # noqa: BLE001
        return None


def _prepare_single_ligand(
    args: tuple[int, Path, Path],
) -> Path | PoseFailure:
    """Convert one pose PDB to PDBQT.

    Routes phospho-residue poses (TPO/SEP/PTR) through Meeko's Polymer API,
    which natively handles phosphate group atom types and Gasteiger charges.
    Standard peptides go through the babel path with a sanitization pre-pass.

    This function is intentionally at module level to satisfy ProcessPoolExecutor
    pickling requirements. Do not move it inside prepare_ligand_batch.

    PDB sanitization (always applied):
      Crystal-pose PDBs from AMBER/CHARMM prep tools contain ionisation-state
      markers in the element column (N1+, O1-, …).  babel uses that column to
      determine element type; unrecognised strings cause the whole molecule to be
      labelled ``UNL`` (unknown ligand), corrupting Gasteiger charge assignment
      for every atom.  ``sanitize_pdb_for_babel`` strips these markers before
      invoking babel so that residue names are preserved correctly.

    UNL detection (post-babel QC):
      After babel runs, the fraction of ATOM/HETATM records labelled ``UNL`` is
      checked.  If it exceeds ``_UNL_FRACTION_THRESHOLD`` a WARNING is logged.
      The PDBQT is still returned — UNL does not always corrupt the score
      materially — but the warning flags the pose for manual review.  Common
      causes: ACE/NME capping groups, non-standard residue names, missing
      CONECT records in a crystal-derived PDB.

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
    pdb_path = Path(pdb_path)
    output_dir = Path(output_dir)

    # Phospho-residue fast path — Meeko handles TPO/SEP/PTR natively.
    from hybridock_pep.prep.phospho import has_phospho_residues, prepare_phospho_ligand
    if has_phospho_residues(pdb_path):
        logger.debug("Pose %d has phospho residues — routing through Meeko", pose_idx)
        return prepare_phospho_ligand(pose_idx, pdb_path, output_dir)

    pdbqt_path = output_dir / (pdb_path.stem + ".pdbqt")

    try:
        babel_bin = shutil.which("babel")
        if babel_bin is None:
            raise FileNotFoundError(
                "babel not found on PATH — install ADFRsuite and add its bin/ to PATH"
            )

        # Sanitize element column before invoking babel (strips N1+, O1-, etc.)
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            clean_pdb = Path(tmp.name)
        try:
            sanitize_pdb_for_babel(pdb_path, clean_pdb)
            result = subprocess.run(
                [babel_bin, "-i", "pdb", str(clean_pdb), "-o", "pdbqt", str(pdbqt_path),
                 "-h", "-xr"],
                capture_output=True,
                text=True,
            )
        finally:
            clean_pdb.unlink(missing_ok=True)

        # babel exits 0 and creates an empty file on input errors — check both
        if result.returncode != 0 or not pdbqt_path.exists() or pdbqt_path.stat().st_size == 0:
            raise RuntimeError(
                f"babel exited {result.returncode} with empty/missing output: "
                f"{result.stderr.strip()}"
            )
        _wrap_pdbqt_rigid(pdbqt_path)

        # UNL detection: babel falls back to "unknown ligand" when it cannot
        # parse residue names (N1+/O1- element column markers, unusual atom
        # names, multi-chain crystal poses, etc.).  UNL corrupts Gasteiger
        # charges across the whole molecule — charge propagation treats all N
        # residues as one connected small molecule rather than per-residue.
        unl_frac = _count_unl_fraction(pdbqt_path)
        if unl_frac > _UNL_FRACTION_THRESHOLD:
            # Try ADFRsuite prepare_ligand4.py as fallback — it uses AutoDockTools
            # molecule parsing which correctly identifies amino-acid residues and
            # assigns per-residue Gasteiger charges even for problematic crystal PDBs.
            fallback_pdbqt = _try_prepare_ligand4_fallback(pdb_path, pdbqt_path)
            if fallback_pdbqt is not None:
                logger.warning(
                    "Pose %d: %.0f%% UNL from babel — used prepare_ligand4.py fallback "
                    "for correct Gasteiger charges (%s)",
                    pose_idx, unl_frac * 100, pdb_path.name,
                )
                return fallback_pdbqt
            else:
                logger.warning(
                    "Pose %d: %.0f%% of PDBQT atoms labelled UNL — babel failed to "
                    "parse residue names in %s and prepare_ligand4.py fallback "
                    "unavailable. Gasteiger charges may be unreliable for AD4 scoring. "
                    "Cause: non-standard crystal-pose PDB (N1+/O1- element markers, "
                    "unusual atom names, or multi-chain structure).",
                    pose_idx, unl_frac * 100, pdb_path.name,
                )

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
