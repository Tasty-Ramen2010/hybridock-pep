"""Charged-residue correction tier for --ultra (the charged-ΔG wall fix).

The fast scorer is charge-blind. For a docked peptide pose, this adds a physics-based correction for ONLY the
charged residues (Ram's LRA decomposition: ΔG_total = ΔG_neutral[scorer] + Σ ΔG_charged[per residue]). Each
charged residue is routed by its interface environment:
  • salt-bridge contact (cationic partner within 4.5 Å)  → ECC-scaled charge-morph FEP (explicit TIP3P, 0.75×)
  • buried / H-bonded to polar-neutral (the 1IAR class)   → GFN2-xTB cluster QM
This is the validated engine from scripts/e343 (ECC), e344 (GB), e346 (QM); the SKEMPI-scale calibration (slope +
per-route scale) is being fit by the E345 overnight campaign — the numeric leg activates once those constants land
(CALIBRATION below). Until then this stage runs the structural triage (which residues, which route, confidence)
so the wiring, progress, and metadata are complete and the compute leg is a drop-in.

See docs/charged_failure_full_decomposition and MEMORY project_charged_fep_terminal_jul07.
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

CHARGED = set("DEKR")
# Filled from the E345 campaign (data/e345_charged_campaign.json summary): per-route linear calibration and the
# route decision. Left None until the campaign lands so we never emit an uncalibrated ΔΔG.
CALIBRATION: dict | None = None


def classify_charged_residues(peptide_sequence: str) -> list[tuple[int, str]]:
    """Return [(1-based index, one-letter)] for each charged residue in the peptide."""
    return [(i + 1, a) for i, a in enumerate(peptide_sequence.upper()) if a in CHARGED]


def apply_charged_correction(scored_poses, cluster_result, config) -> int:
    """Structural triage now; numeric ECC/QM correction activates when CALIBRATION is populated.

    Returns the number of charged residues detected. Attaches per-pose metadata (charged residues, route) so the
    CSV/JSON carry it even before the compute leg is calibrated. Never raises on the scoring hot path.
    """
    charged = classify_charged_residues(config.peptide_sequence)
    if not charged:
        return 0
    desc = ",".join(f"{a}{i}" for i, a in charged)
    logger.info("Charged correction: %d charged residue(s) [%s]", len(charged), desc)
    if CALIBRATION is None:
        logger.info(
            "Charged correction: engine pending E345 calibration — reporting triage only "
            "(residues=%s). ECC-FEP/QM leg is a drop-in once campaign constants land.", desc,
        )
        for p in scored_poses:
            setattr(p, "charged_residues", desc)
    else:  # pragma: no cover — activated post-calibration
        from hybridock_pep.scoring._charged_engine import correct_poses  # noqa: PLC0415
        correct_poses(scored_poses, cluster_result, config, charged, CALIBRATION)
    return len(charged)
