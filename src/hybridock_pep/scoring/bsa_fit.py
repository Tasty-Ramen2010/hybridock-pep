"""BSA-fit pose ranker — buried surface area with a clash penalty.

Replaces ref2015 as the *pose ranker* (which of the diffusion poses is the
tightest, most physically valid fit). Validated on 112 bench300 complexes: a
BSA-plus-clash fit score ties ref2015 for selected-pose RMSD (~4.5 Å mean,
~47% CAPRI-acceptable) at a fraction of the compute cost and with no PyRosetta
dependency — and higher BSA genuinely tracks lower RMSD (the tightest valid
pose IS the native one, thermodynamically). See docs/scoring_overhaul_verdict.md
and the pose-selection test (scripts/pipeline_selection_test.py).

Score (lower = better fit, to match the ascending rank convention):

    bsa_fit_score = -z(BSA) + clash_weight * z(n_clash)

where BSA is the interface buried surface area (Å²) and n_clash is the number of
peptide-receptor heavy-atom pairs closer than CLASH_DIST (steric overlap). Both
are z-normalised *within the current pose set* (one receptor, N poses), so the
score is a pure within-complex ranking signal. Raw BSA and clash counts are also
stored on the pose for transparency.

This does NOT touch the affinity / Kd number (Vina / hybrid_score stay as-is).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from hybridock_pep.models import ScoredPose

logger = logging.getLogger(__name__)

CLASH_DIST: float = 3.0          # Å heavy-atom overlap → steric clash
CROP_RADIUS: float = 10.0        # Å around peptide for the receptor SASA crop
_DEFAULT_CLASH_WEIGHT: float = 1.0


def _read_heavy(pdb: Path) -> tuple[list[str], np.ndarray]:
    """Return (atom_lines, xyz) for heavy atoms of a PDB."""
    lines: list[str] = []
    xyz: list[tuple[float, float, float]] = []
    for ln in pdb.read_text().splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        an = ln[12:16].strip()
        if not an or an[0] in ("H", "D"):
            continue
        try:
            xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
        except ValueError:
            continue
        lines.append(ln)
    return lines, np.array(xyz) if xyz else np.empty((0, 3))


def _sasa(lines: list[str]) -> float:
    """Shrake-Rupley total SASA of a set of PDB ATOM lines (Biopython)."""
    import io
    import warnings

    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley

    if not lines:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        struct = PDBParser(QUIET=True).get_structure("x", io.StringIO("\n".join(lines) + "\nEND\n"))
        ShrakeRupley().compute(struct, level="A")
        return float(sum(a.sasa for a in struct.get_atoms()))


def compute_bsa_fit_scores(
    poses: list[ScoredPose],
    receptor_pdb: Path,
    clash_weight: float = _DEFAULT_CLASH_WEIGHT,
) -> None:
    """Set ``bsa_fit_score`` (and raw ``bsa`` / ``n_clash``) on each pose in place.

    All poses must share the one receptor (a single dock run). Receptor SASA is
    cropped to within ``CROP_RADIUS`` of each pose's peptide for speed; the
    cropped receptor's own SASA is recomputed per pose because the crop set
    depends on the pose. Failures leave ``bsa_fit_score`` = None for that pose.

    Args:
        poses: Scored poses from one receptor; mutated in place.
        receptor_pdb: Path to the (full) receptor PDB.
        clash_weight: Weight on the z-normalised clash penalty. Default 1.0.
    """
    rec_lines, rec_xyz = _read_heavy(receptor_pdb)
    if len(rec_xyz) == 0:
        logger.warning("BSA-fit: receptor %s has no heavy atoms; skipping", receptor_pdb)
        return

    bsas: list[float | None] = []
    clashes: list[float | None] = []
    for pose in poses:
        try:
            pep_lines, pep_xyz = _read_heavy(pose.pdb_path)
            if len(pep_xyz) == 0:
                raise ValueError("pose has no heavy atoms")
            # crop receptor to atoms within CROP_RADIUS of any peptide atom
            d2 = ((rec_xyz[:, None, :] - pep_xyz[None, :, :]) ** 2).sum(-1)
            near = d2.min(1) <= CROP_RADIUS ** 2
            crop_lines = [rec_lines[i] for i in np.where(near)[0]]
            s_pep = _sasa(pep_lines)
            s_rec = _sasa(crop_lines)
            s_cx = _sasa(crop_lines + pep_lines)
            bsa = s_pep + s_rec - s_cx
            n_clash = float((d2.min(1) < CLASH_DIST ** 2).sum())  # peptide atoms overlapping
            pose.bsa = bsa
            pose.n_clash = n_clash
            bsas.append(bsa)
            clashes.append(n_clash)
        except Exception as exc:  # noqa: BLE001 — per-pose robustness
            logger.warning("BSA-fit failed for pose %d (%s)", pose.pose_idx, exc)
            bsas.append(None)
            clashes.append(None)

    valid_b = [b for b in bsas if b is not None]
    valid_c = [c for c in clashes if c is not None]
    if len(valid_b) < 2:
        logger.warning("BSA-fit: <2 valid poses; cannot z-normalise, leaving scores None")
        return
    b_mu, b_sd = float(np.mean(valid_b)), float(np.std(valid_b)) or 1.0
    c_mu, c_sd = float(np.mean(valid_c)), float(np.std(valid_c)) or 1.0

    for pose, b, c in zip(poses, bsas, clashes):
        if b is None or c is None:
            continue
        z_bsa = (b - b_mu) / b_sd
        z_clash = (c - c_mu) / c_sd
        # lower = better fit (tighter burial, fewer clashes) → ascending rank
        pose.bsa_fit_score = -z_bsa + clash_weight * z_clash

    logger.info(
        "BSA-fit ranked %d/%d poses (BSA mean=%.0f Å², clash mean=%.1f, clash_w=%.1f)",
        sum(1 for p in poses if p.bsa_fit_score is not None), len(poses),
        b_mu, c_mu, clash_weight,
    )
