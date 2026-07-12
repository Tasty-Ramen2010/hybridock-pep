"""Free-state conformational entropy feature for the ensemble scorer (SCORE-ENT).

A peptide loses conformational entropy when it freezes on binding (+TΔS penalty). This term is
INVISIBLE to a static bound pose — it depends on the unbound peptide's flexibility — so we
estimate it from a short MD simulation of the FREE peptide and measure its dihedral-histogram
entropy S_free. Validated (docs E40): adding s_free_bur to the geometry+entropy model lifts
pooled cross-dataset LOO from 0.409 to 0.488 (permutation-validated), the first feature to
meaningfully improve diverse generalization.

Physics: floppy peptides (high S_free) bind weaker; pre-rigid peptides (poly-Pro, β-branched,
disulfide) pay little entropy. s_free_bur = S_free × buried_fraction weights it by how much of
the peptide actually freezes at the interface.

Cost: ~8 s/peptide on GPU (60 ps free-peptide MD via OpenMM/GBn2). Opt-in; requires OpenMM.
Reuses the MD machinery in experiments/e18v2_md.run_free_dynamics (ff14SB + GBn2 implicit solvent).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def compute_free_state_entropy(peptide_pdb: Path, prod_ps: int = 60) -> dict | None:
    """Run free-peptide MD and return free-state conformational entropy descriptors.

    Args:
        peptide_pdb: PDB of the peptide alone (any conformation; the MD relaxes it).
        prod_ps: Production MD length in picoseconds (default 60; ~8 s on GPU).

    Returns:
        Dict with:
          s_free       — mean per-residue dihedral-histogram entropy (nats), the free-state
                         conformational entropy lost on binding (higher → weaker binding).
          s_free_total — summed over residues (extensive).
          rmsf         — mean Cα RMSF (Å), backbone mobility of the free peptide.
        Returns None if the MD diverges or OpenMM is unavailable.
    """
    import sys

    # The MD helpers live in scripts/ (shared with the research pipeline). Add to path lazily so
    # importing this module never hard-requires the scripts dir or OpenMM at import time.
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        from e18v2_md import run_free_dynamics  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        logger.warning("free_entropy: MD machinery unavailable (%s)", exc)
        return None

    try:
        rmsf, s_dih = run_free_dynamics(str(peptide_pdb), prod_ps)
    except Exception as exc:  # noqa: BLE001
        logger.warning("free_entropy: free MD failed for %s (%s)", peptide_pdb, exc)
        return None

    s_free = float(np.nanmean(s_dih))
    if not np.isfinite(s_free):
        return None
    return dict(
        s_free=s_free,
        s_free_total=float(np.nansum(s_dih)),
        rmsf=float(np.nanmean(rmsf)),
    )


def s_free_buried(s_free: float, buried_fraction: float) -> float:
    """The entropy that actually freezes: S_free weighted by interface burial.

    Args:
        s_free: Free-state mean dihedral entropy from compute_free_state_entropy.
        buried_fraction: Fraction of the peptide buried at the interface (0..1), e.g.
            geometry_features f_hyd_iface.

    Returns:
        s_free × clamp(buried_fraction, 0, 1) — the validated ensemble feature (docs E40).
    """
    return float(s_free) * float(min(1.0, max(0.0, buried_fraction)))
