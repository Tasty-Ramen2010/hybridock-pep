from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from openmm.app import PDBFile
from pdbfixer import PDBFixer

from hybridock_pep.models import DockConfig
from hybridock_pep.prep.errors import PrepError

logger = logging.getLogger(__name__)


def prepare_receptor(config: DockConfig) -> Path:
    """Clean a receptor PDB with pdbfixer and convert it to PDBQT via prepare_receptor4.py.

    Always regenerates the PDBQT — no caching, no mtime checks (D-02).
    pdbfixer steps run unconditionally (D-01):
      1. Strip non-water HETATM and alternate-occupancy atoms (keep alt ' ' or 'A').
      2. Find and add missing residues.
      3. Find and add missing atoms.
      4. Add hydrogens at pH 7.4.

    If prepare_receptor4.py exits non-zero, raises PrepError immediately with the
    full stderr captured (D-03). No retry, no fallback.

    Args:
        config: Validated DockConfig. Uses receptor_path and output_dir.

    Returns:
        Path to the written receptor PDBQT (output_dir/receptor.pdbqt).

    Raises:
        PrepError: If prepare_receptor4.py exits non-zero.
        FileNotFoundError: If prepare_receptor4.py is not on PATH.
    """
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pdbqt_path = output_dir / "receptor.pdbqt"

    # --- Step 1: Pre-filter PDB (strip altLoc B/C/... and non-water HETATM) ---
    cleaned_pdb_lines = _filter_pdb_lines(config.receptor_path)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
        tmp.writelines(cleaned_pdb_lines)
        cleaned_pdb_path = Path(tmp.name)

    try:
        # --- Step 2: pdbfixer — all three fixes, unconditionally (D-01) ---
        fixer = PDBFixer(filename=str(cleaned_pdb_path))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingHydrogens(7.4)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pdb", delete=False
        ) as fixed_tmp:
            PDBFile.writeFile(fixer.topology, fixer.positions, fixed_tmp)
            fixed_pdb_path = Path(fixed_tmp.name)
    finally:
        cleaned_pdb_path.unlink(missing_ok=True)

    # --- Step 3: prepare_receptor4.py (always regenerate — D-02) ---
    cmd = [
        "prepare_receptor4.py",
        "-r", str(fixed_pdb_path),
        "-o", str(pdbqt_path),
    ]
    logger.info("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
    finally:
        fixed_pdb_path.unlink(missing_ok=True)

    if result.returncode != 0:
        raise PrepError(
            f"prepare_receptor4.py failed (exit {result.returncode}):\n{result.stderr}"
        )

    logger.info("Receptor PDBQT written: %s", pdbqt_path)
    return pdbqt_path


def _filter_pdb_lines(pdb_path: Path) -> list[str]:
    """Strip alternate-occupancy atoms and non-water HETATM from PDB text.

    Keeps ATOM records and water HETATM (resName HOH or WAT) where altLoc is
    blank (' ') or 'A'. All other records (REMARK, HEADER, etc.) are passed through.

    Args:
        pdb_path: Path to the input PDB file.

    Returns:
        List of filtered PDB lines, each ending with newline.
    """
    kept: list[str] = []
    for line in pdb_path.read_text().splitlines(keepends=True):
        record = line[:6].strip()
        if record == "HETATM":
            res_name = line[17:20].strip()
            if res_name not in ("HOH", "WAT"):
                continue  # drop non-water HETATM
        if record in ("ATOM", "HETATM"):
            alt_loc = line[16] if len(line) > 16 else " "
            if alt_loc not in (" ", "A"):
                continue  # drop alternate occupancy B/C/...
        kept.append(line)
    return kept
