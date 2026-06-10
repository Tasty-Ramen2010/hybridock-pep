"""NIS (non-interacting surface) composition — within-target relative affinity.

**What this is.** The polar/charged composition of a peptide's *non-contacting*
residues — PRODIGY's non-interacting-surface idea, ported to peptides. Validated
this session (``scripts/e0_*`` … ``scripts/e2_*``;
``docs/kcalmol_research_synthesis.md``) as the one correctly-signed, BSA-orthogonal
feature that tracks binding affinity.

**What it is for — and what it is NOT.** NIS carries genuine signal *within a
target* (ranking peptide variants against one receptor, Pearson ≈ 0.4) but ≈ 0
*across* protein families (one-per-family r ≈ 0.07). It is therefore a **relative**
signal — rank variants / take a ΔΔG against a reference binder — **not** an
absolute cross-target kcal/mol predictor. Every cheap absolute-ΔG number on
peptides is the interface-size confound (see the verdict doc); NIS is the honest
relative alternative. This module deliberately does **not** touch the affinity /
hybrid_score number.

Per pose (peptide vs one receptor):

    nis_polar_frac   = (# non-contacting peptide residues that are polar)   / N_nis
    nis_charged_frac = (# non-contacting peptide residues that are charged) / N_nis
    nis_score        = nis_charged_frac − nis_polar_frac

``nis_score`` is oriented so that **lower = stronger predicted binding** (matching
the ΔG sign and the ascending-rank convention used by ``bsa_fit``). For variant
ranking it is z-normalised within the candidate set by
:func:`relative_nis_ranking`.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np

from hybridock_pep.models import ScoredPose

logger = logging.getLogger(__name__)

CONTACT_CUT: float = 5.5  # Å heavy-atom cutoff defining an interfacial residue

# Chemical classes (His counted charged, as in the validation).
_CHARGED = {"ARG", "LYS", "ASP", "GLU", "HIS"}
_POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "TRP", "HIS"}


def _cls(resname: str) -> str:
    rn = resname.upper()
    if rn in _CHARGED:
        return "C"
    if rn in _POLAR:
        return "P"
    return "A"


def compute_nis_features(
    peptide_pdb: Path,
    receptor_pdb: Path,
    contact_cut: float = CONTACT_CUT,
) -> tuple[float, float]:
    """Return ``(nis_polar_frac, nis_charged_frac)`` for one peptide/receptor pair.

    A peptide residue is *interfacial* if any of its heavy atoms is within
    ``contact_cut`` Å of a receptor heavy atom; the rest form the
    non-interacting surface (NIS), whose polar/charged residue fractions are
    returned.

    Args:
        peptide_pdb: Path to the peptide pose PDB.
        receptor_pdb: Path to the receptor PDB (one receptor).
        contact_cut: Heavy-atom contact cutoff in Å.

    Returns:
        ``(nis_polar_frac, nis_charged_frac)``, each in [0, 1]. ``(0.0, 0.0)`` if
        the peptide has no non-interacting residues (fully buried) or parsing
        yields no residues.

    Raises:
        FileNotFoundError: If either PDB path does not exist.
    """
    if not Path(peptide_pdb).exists():
        raise FileNotFoundError(peptide_pdb)
    if not Path(receptor_pdb).exists():
        raise FileNotFoundError(receptor_pdb)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from Bio.PDB import NeighborSearch, PDBParser

        parser = PDBParser(QUIET=True)
        pep = parser.get_structure("pep", str(peptide_pdb))
        rec = parser.get_structure("rec", str(receptor_pdb))

    pep_res = [r for r in pep[0].get_residues() if r.id[0] == " "]
    rec_atoms = [a for r in rec[0].get_residues() if r.id[0] == " "
                 for a in r if a.element != "H"]
    if not pep_res or not rec_atoms:
        return 0.0, 0.0

    ns = NeighborSearch(rec_atoms)
    cut2 = contact_cut
    n_polar = n_charged = n_nis = 0
    for rp in pep_res:
        contacting = any(ns.search(a.coord, cut2) for a in rp if a.element != "H")
        if contacting:
            continue
        n_nis += 1
        cl = _cls(rp.resname)
        n_polar += cl == "P"
        n_charged += cl == "C"

    if n_nis == 0:
        return 0.0, 0.0
    return n_polar / n_nis, n_charged / n_nis


def compute_nis_scores(
    poses: list[ScoredPose],
    receptor_pdb: Path,
    contact_cut: float = CONTACT_CUT,
) -> None:
    """Set ``nis_polar_frac`` / ``nis_charged_frac`` / ``nis_score`` on each pose.

    All poses must share the one receptor. ``nis_score = nis_charged_frac −
    nis_polar_frac`` is the raw (un-normalised) oriented value; lower = stronger
    predicted binding. Per-pose failures leave the fields ``None``.

    Args:
        poses: Scored poses from one receptor; mutated in place.
        receptor_pdb: Path to the receptor PDB.
        contact_cut: Heavy-atom contact cutoff in Å.
    """
    for pose in poses:
        try:
            polar, charged = compute_nis_features(pose.pdb_path, receptor_pdb, contact_cut)
            pose.nis_polar_frac = polar
            pose.nis_charged_frac = charged
            pose.nis_score = charged - polar
        except Exception as exc:  # noqa: BLE001 — per-pose robustness
            logger.warning("NIS failed for pose %d (%s)", pose.pose_idx, exc)
            pose.nis_polar_frac = None
            pose.nis_charged_frac = None
            pose.nis_score = None


def relative_nis_ranking(nis_scores: list[float]) -> np.ndarray:
    """Z-normalise raw ``nis_score`` values across a candidate set (variants).

    Use this to rank peptide variants against ONE receptor — the validated
    within-target regime. Lower (more negative z) = stronger predicted binding.

    Args:
        nis_scores: Raw ``nis_score`` per candidate (``nis_charged − nis_polar``).

    Returns:
        Z-normalised scores (mean 0, sd 1). If sd is 0, returns zeros.

    Raises:
        ValueError: If fewer than two scores are provided (no relative frame).
    """
    arr = np.asarray(nis_scores, dtype=float)
    if arr.size < 2:
        raise ValueError("relative NIS ranking needs >= 2 candidates")
    sd = float(arr.std())
    if sd < 1e-9:  # constant (within float dust) -> no relative frame
        return np.zeros_like(arr)
    return (arr - float(arr.mean())) / sd
