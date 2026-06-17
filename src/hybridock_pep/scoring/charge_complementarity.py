"""Within-target charge complementarity — a salt-bridge / electrostatic-complementarity score for ranking
peptides against a SINGLE receptor (selectivity), NOT for cross-target absolute Kd.

Why this exists (the hard-won boundary, E240-E256):
  Charged absolute Kd is FEP-bound. We proved it exhaustively: a salt bridge's net ΔG is a small difference
  of large cancelling terms (Coulomb vs desolvation), and the per-receptor "offset" that sets the absolute
  value is NOT predictable from any static OR single-MD representation (sequence, ESM, fpocket, Poisson-
  Boltzmann, and 0.6 ns explicit-water GIST all <= permutation null on the offset). Computing it needs a
  cross-state free-energy difference = FEP.

  BUT the WITHIN-target signal is real and strong. When ranking peptides against ONE pocket, the receptor
  offset CANCELS (it is shared), exposing the charge-charge interaction physics. On SKEMPI charge-changing
  mutations this within-pocket signal reaches r=0.755 with a known offset (E254 ceiling) and the
  pure interaction model transfers at r~0.28-0.66 pooled. That is the selectivity regime — exactly our
  deployment use case and the frame PPI-Affinity cannot run (no pose generation).

This module scores the charge complementarity of a peptide pose against the receptor's charged groups:
  favorable opposite-charge contacts (salt bridges) - like-charge repulsion, distance-screened. It is a
  RANKING signal for one receptor; do not interpret the magnitude as an absolute ΔG.
"""
from __future__ import annotations

import numpy as np

# formal side-chain charge at pH 7
_QSIGN = {"K": 1.0, "R": 1.0, "H": 0.1, "D": -1.0, "E": -1.0}
_COULOMB_K = 332.0  # kcal·Å / (mol·e²), with eps folded into the screening below


def _charged_atoms(residues: list[tuple[str, np.ndarray]]) -> list[tuple[float, np.ndarray]]:
    """(charge, sidechain-tip coord) for each charged residue in a (resletter, coord) list."""
    return [(_QSIGN[aa], xyz) for aa, xyz in residues if aa in _QSIGN and abs(_QSIGN[aa]) > 0.5]


def charge_complementarity_score(
    peptide_groups: list[tuple[str, np.ndarray]],
    receptor_groups: list[tuple[str, np.ndarray]],
    *,
    cutoff: float = 12.0,
) -> dict[str, float]:
    """Distance-screened charge complementarity of a peptide pose vs receptor charged groups.

    Args:
        peptide_groups: list of (residue one-letter, side-chain-tip xyz Å) for the docked peptide.
        receptor_groups: same for the receptor's charged residues.
        cutoff: ignore pairs beyond this distance (Å). Screening uses a distance-dependent dielectric
            (eps = 4r) consistent with the within-pocket validation; 1/r² net form.

    Returns:
        Dict of ranking features (for one receptor):
          ``salt_bridge``     screened sum of OPPOSITE-charge pair energies (favorable, negative)
          ``repulsion``       screened sum of LIKE-charge pair energies (unfavorable, positive)
          ``net_elec``        salt_bridge + repulsion (the net interaction signal)
          ``n_salt_bridges``  count of opposite-charge pairs within 6 Å
          ``n_repulsive``     count of like-charge pairs within 6 Å

    Note:
        This is a WITHIN-target ranking signal (selectivity). It is NOT calibrated to absolute kcal/mol and
        must not be summed into a cross-target ΔG (the receptor offset it omits is FEP-only).
    """
    pep = _charged_atoms(peptide_groups)
    rec = _charged_atoms(receptor_groups)
    salt = 0.0
    rep = 0.0
    n_sb = 0
    n_rep = 0
    for qp, xp in pep:
        for qr, xr in rec:
            r = float(np.linalg.norm(xp - xr))
            if r < 1.5 or r > cutoff:
                continue
            e = _COULOMB_K * qp * qr / (4.0 * r * r)  # eps=4r screened, 1/r² net form
            if qp * qr < 0:        # opposite charges → salt bridge (favorable, e<0)
                salt += e
                if r < 6.0:
                    n_sb += 1
            else:                  # like charges → repulsion (unfavorable, e>0)
                rep += e
                if r < 6.0:
                    n_rep += 1
    return {
        "salt_bridge": salt,
        "repulsion": rep,
        "net_elec": salt + rep,
        "n_salt_bridges": float(n_sb),
        "n_repulsive": float(n_rep),
    }
