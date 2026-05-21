"""Meeko-based PDBQT preparation for peptides containing phospho-residues.

Handles TPO (phospho-Thr), SEP (phospho-Ser), and PTR (phospho-Tyr).
Meeko 0.7+ ships built-in templates for all three residues, so no custom
template construction is needed.

Called by prep/ligand.py when a pose PDB contains any of these residue names.
Standard (non-phospho) peptides continue through the babel path unchanged.
"""
from __future__ import annotations

import logging
from pathlib import Path

from hybridock_pep.models import PoseFailure

logger = logging.getLogger(__name__)

PHOSPHO_RESIDUES: frozenset[str] = frozenset({"TPO", "SEP", "PTR"})


def has_phospho_residues(pdb_path: Path) -> bool:
    """Return True if the PDB file contains any phospho-residue ATOM/HETATM record."""
    try:
        text = pdb_path.read_text(errors="replace")
    except OSError:
        return False
    for line in text.splitlines():
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 20:
            resname = line[17:20].strip()
            if resname in PHOSPHO_RESIDUES:
                return True
    return False


def prepare_phospho_ligand(
    pose_idx: int,
    pdb_path: Path,
    output_dir: Path,
) -> Path | PoseFailure:
    """Convert a phospho-containing peptide PDB to PDBQT via Meeko.

    Uses Meeko's Polymer API which natively handles TPO/SEP/PTR residues.
    Assigns Gasteiger partial charges (same as babel -h) including the
    phosphate group, which babel silently drops or mis-types.

    Args:
        pose_idx: Zero-based pose index (for error reporting).
        pdb_path: Path to the pose PDB to convert.
        output_dir: Directory to write the PDBQT file.

    Returns:
        Path to the written PDBQT on success, or PoseFailure on error.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pdbqt_path = output_dir / (pdb_path.stem + ".pdbqt")

    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy, Polymer, ResidueChemTemplates

        pdb_text = pdb_path.read_text(errors="replace")
        rct = ResidueChemTemplates.create_from_defaults()
        mk_prep = MoleculePreparation(
            merge_these_atom_types=("H",),
            charge_model="gasteiger",
        )
        polymer = Polymer.from_pdb_string(
            pdb_text,
            rct,
            mk_prep,
            allow_bad_res=False,
        )
        polymer.parameterize(mk_prep)
        pdbqt_string, _ = PDBQTWriterLegacy.write_string_from_polymer(polymer)

        if not pdbqt_string.strip():
            raise ValueError("Meeko produced empty PDBQT output")

        # Vina 1.2.x requires ROOT/ENDROOT/TORSDOF wrapper — same requirement as babel output.
        # Meeko's Polymer writer outputs a flat PDBQT without the tree structure.
        lines = pdbqt_string.splitlines(keepends=True)
        remarks = [l for l in lines if l.startswith("REMARK")]
        atoms = [l for l in lines if l.startswith(("ATOM", "HETATM"))]
        wrapped = "".join(remarks) + "ROOT\n" + "".join(atoms) + "ENDROOT\nTORSDOF 0\n"

        pdbqt_path.write_text(wrapped)
        logger.debug("Phospho PDBQT written: %s", pdbqt_path)
        return pdbqt_path

    except Exception as exc:
        return PoseFailure(
            pose_idx=pose_idx,
            stage="prep_phospho",
            error_msg=f"{type(exc).__name__}: {exc}",
        )
