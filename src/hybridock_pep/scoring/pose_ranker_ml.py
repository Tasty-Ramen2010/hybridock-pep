"""ML pose ranker — predicts native Cα-RMSD from computable, OSI-clean pose features.

This ranks the diffusion poses for the STRUCTURAL deliverable (``best_pose.pdb`` and the
ranked-CSV order). It does NOT touch the affinity / ΔG number — selecting near-native poses
*hurts* cross-complex affinity r (the pocket, not the pose geometry, carries Kd; see E94 /
docs/DEVELOPMENT_TIMELINE.md). Keep this wall: ``ml_pose_score`` feeds pose ORDERING only.

Validated head-to-head on 46 real RAPiDock complexes, leave-one-complex-out, within-complex
Kendall τ vs native RMSD (920 poses, scripts/e96_poseranker_validation.py):

    production BSA+clash ranker : τ = +0.201
    this ML ranker (computable) : τ = +0.406   (≈2× better)

Features (all computable from the pose alone — no Rescore+, no mordred, OSI/MIT-clean):
  * Ramachandran family — region fractions + φ/ψ means + φ/ψ KDE log-probabilities. KDEs are
    fitted at train time and bundled (no external dependency at runtime).
  * 3D-shape family — RDKit PMI1/2/3, NPR1/2, asphericity, eccentricity, radius-of-gyration,
    inertial-shape-factor, spherocity, plane-best-fit (PBF).

Approach adapted from PepScorerRMSD (A. G. Cavalli, 2025, MIT) — its non-OSI Rescore+ terms
(PLANTS/APBS/CHARMM/X-Score) are deliberately omitted; only the computable subset is used.

The model artifact lives at ``data/pose_ranker_ml.joblib`` (built by
``scripts/train_pose_ranker_ml.py``). If it is missing or any dependency fails, scoring is a
silent no-op (``ml_pose_score`` stays None) and ranking falls back to BSA-fit — so the tool
never crashes on a fresh install that hasn't built the artifact.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from hybridock_pep.models import ScoredPose

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2].parent / "data" / "pose_ranker_ml.joblib"

RAMA_NAMES = ["rama_reg1", "rama_reg2", "rama_reg3", "rama_reg4",
              "phi_mean", "psi_mean", "phi_logp", "psi_logp"]
SHAPE_NAMES = ["pmi1", "pmi2", "pmi3", "npr1", "npr2", "asphericity", "eccentricity",
               "radius_gyration", "inertial_shape", "spherocity", "pbf"]
FEATURE_NAMES = RAMA_NAMES + SHAPE_NAMES


def rama_features(pose_pdb: Path, phi_kde: Any, psi_kde: Any) -> list[float] | None:
    """Ramachandran region fractions + φ/ψ means + φ/ψ KDE log-probabilities.

    Args:
        pose_pdb: Path to a peptide pose PDB.
        phi_kde: Fitted scipy ``gaussian_kde`` over training φ angles (degrees).
        psi_kde: Fitted scipy ``gaussian_kde`` over training ψ angles (degrees).

    Returns:
        Eight-element feature list, or None if no φ/ψ pair could be computed.
    """
    from Bio.PDB import PDBParser, Polypeptide

    struct = PDBParser(QUIET=True).get_structure("x", str(pose_pdb))[0]
    reg = [0, 0, 0, 0]
    phis: list[float] = []
    psis: list[float] = []
    for chain in struct:
        try:
            poly = Polypeptide.Polypeptide(chain)
        except Exception:  # noqa: BLE001 — malformed chain; skip
            continue
        for residue in poly.get_phi_psi_list():
            if None in residue:
                continue
            phi, psi = math.degrees(residue[0]), math.degrees(residue[1])
            phis.append(phi)
            psis.append(psi)
            if ((-130 < phi < -50) and (120 < psi < 180)) or ((-75 < phi < -60) and (-50 < psi < -25)):
                reg[0] += 1
            elif ((-150 < phi < -45) and (100 < psi < 180)) or ((-90 < phi < -45) and (-65 < psi < 0)):
                reg[1] += 1
            elif (-180 < phi < -30) or ((30 < phi < 105) and (-30 < psi < 90)):
                reg[2] += 1
            else:
                reg[3] += 1
    n = len(phis)
    if n == 0:
        return None
    phi_logp = float(np.log(np.mean(phi_kde.evaluate(phis)) + 1e-12))
    psi_logp = float(np.log(np.mean(psi_kde.evaluate(psis)) + 1e-12))
    return [reg[0] / n, reg[1] / n, reg[2] / n, reg[3] / n,
            float(np.mean(phis)), float(np.mean(psis)), phi_logp, psi_logp]


def shape_features(pose_pdb: Path) -> list[float] | None:
    """RDKit 3D-shape descriptors of the peptide pose. None if RDKit can't parse it."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors3D, rdMolDescriptors

    mol = Chem.MolFromPDBFile(str(pose_pdb), sanitize=False, removeHs=True)
    if mol is None or mol.GetNumConformers() == 0:
        return None
    try:
        mol.UpdatePropertyCache(strict=False)
        return [
            rdMolDescriptors.CalcPMI1(mol), rdMolDescriptors.CalcPMI2(mol), rdMolDescriptors.CalcPMI3(mol),
            rdMolDescriptors.CalcNPR1(mol), rdMolDescriptors.CalcNPR2(mol),
            Descriptors3D.Asphericity(mol), Descriptors3D.Eccentricity(mol),
            Descriptors3D.RadiusOfGyration(mol), Descriptors3D.InertialShapeFactor(mol),
            Descriptors3D.SpherocityIndex(mol), rdMolDescriptors.CalcPBF(mol),
        ]
    except Exception:  # noqa: BLE001 — degenerate geometry; skip this pose
        return None


def compute_features(pose_pdb: Path, phi_kde: Any, psi_kde: Any) -> list[float] | None:
    """Full computable feature vector (Ramachandran + 3D-shape) for one pose."""
    rf = rama_features(pose_pdb, phi_kde, psi_kde)
    if rf is None:
        return None
    sf = shape_features(pose_pdb)
    if sf is None:
        return None
    return rf + sf


def _load_bundle(model_path: Path) -> dict[str, Any] | None:
    if not model_path.exists():
        logger.info("ML pose ranker artifact not found at %s — falling back to BSA-fit ranking", model_path)
        return None
    try:
        import joblib
        return joblib.load(model_path)
    except Exception as exc:  # noqa: BLE001 — version skew / corrupt artifact
        logger.warning("ML pose ranker artifact failed to load (%s) — falling back to BSA-fit", exc)
        return None


def compute_ml_pose_scores(
    poses: list[ScoredPose],
    model_path: Path | None = None,
) -> bool:
    """Set ``ml_pose_score`` (predicted native Cα-RMSD, Å; lower = more native) on each pose.

    Pure structural ranking signal — does NOT modify any affinity field. Failures (missing
    artifact, RDKit parse error, missing dependency) leave ``ml_pose_score`` = None for the
    affected pose(s); ranking then falls back to BSA-fit. Never raises.

    Args:
        poses: Scored poses from one dock run; mutated in place.
        model_path: Override for the model artifact path (defaults to data/pose_ranker_ml.joblib).

    Returns:
        True if at least one pose received an ``ml_pose_score``, else False.
    """
    bundle = _load_bundle(model_path or DEFAULT_MODEL_PATH)
    if bundle is None:
        return False
    phi_kde, psi_kde, model = bundle["phi_kde"], bundle["psi_kde"], bundle["model"]
    n_ok = 0
    for pose in poses:
        if pose.pdb_path is None:
            continue
        feats = compute_features(Path(pose.pdb_path), phi_kde, psi_kde)
        if feats is None:
            continue
        try:
            pose.ml_pose_score = float(model.predict(np.array([feats]))[0])
            n_ok += 1
        except Exception:  # noqa: BLE001 — predict shouldn't fail, but never crash ranking
            continue
    if n_ok:
        logger.info("ML pose ranker scored %d/%d poses (predicted native RMSD)", n_ok, len(poses))
    return n_ok > 0
