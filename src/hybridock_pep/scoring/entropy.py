"""Interface burial correction and hybrid score calibration.

Implements the HybriDock-Pep hybrid score formula:
    hybrid = vina + beta*(ad4 - vina) + alpha*n_eff_residues

where ``alpha * n_eff_residues`` is the interface burial correction term,
``n_eff_residues`` is either the full peptide length or the contact-residue
count (residues with at least one heavy atom within CONTACT_DIST_ANG of the
receptor), and ``beta`` controls the blending weight of AD4 relative to Vina.

NOTE ON TERMINOLOGY: This module was originally named "entropy" because the
``alpha * n_contact`` term approximates the entropic cost of peptide burial at
the binding interface, following conventions from implicit-solvent literature
where contact number serves as a proxy for solvation entropy. It is a linear
contact-count correction, not a true entropy calculation.

When ``is_ad4_anomaly`` is True (AD4 score > 0), beta is forced to 0 so
the anomalous AD4 signal does not corrupt the hybrid score.

Calibration (alpha, beta) is loaded from a JSON file and validated on every
read (load_calibration raises ValueError for out-of-range values). Fitting
uses scipy L-BFGS-B with hardcoded bounds.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr

from hybridock_pep.models import ScoredPose

_log = logging.getLogger(__name__)

# Contact distance cutoff — must match score_calibration_set.py _CONTACT_CUTOFF.
# Changing this requires re-running Tier 1.3 so calibration and inference use
# the same contact counts.
CONTACT_DIST_ANG: float = 4.5

# Thermodynamic constant: RT at 298 K in kcal/mol (hardcoded in v1).
_RT = 0.592
# pKd → ΔG: ΔG = -RT * ln(10) * pKd  (Kd = 10^-pKd, ΔG = RT*ln(Kd))
_LN10 = math.log(10)

# Alpha bounds used both in optimization and validation.
# Contact-based entropy uses a wider upper bound than full-residue because
# alpha compensates for the smaller effective residue count.
# Lower bound reduced to 0.1 (2026-05-13): PfLDH calibration shows Vina
# score_only is already close to experimental ΔG for this receptor family,
# so the entropy penalty should be minimal (alpha ≈ 0.1–0.2).
_ALPHA_MIN = 0.1
_ALPHA_MAX = 2.0
_BETA_MIN = 0.0
_BETA_MAX = 0.5


def load_calibration(path: Path) -> dict:
    """Load and validate calibration parameters from a JSON file.

    Validates that alpha is within [0.1, 2.0] kcal/mol/contact-residue
    and beta within [0.0, 0.5]. The upper alpha bound is 2.0 (was 1.2)
    to accommodate contact-based entropy where alpha compensates for
    fewer effective residues than full sequence length.

    Args:
        path: Path to the calibration JSON file.

    Returns:
        Dictionary containing calibration data including 'alpha' and 'beta'.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If alpha is outside [0.1, 2.0] or beta is outside [0.0, 0.5].
    """
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    with path.open() as fh:
        cal = json.load(fh)

    try:
        alpha = cal["alpha"]
        beta = cal["beta"]
    except KeyError as exc:
        raise ValueError(
            f"Calibration file {path} is missing required key {exc}. "
            "Re-run calibrate_alpha.py to regenerate a valid calibration file."
        ) from exc

    if not (_ALPHA_MIN <= alpha <= _ALPHA_MAX):
        raise ValueError(
            f"Calibrated α={alpha:.3f} is outside valid range [{_ALPHA_MIN}, {_ALPHA_MAX}] "
            "kcal/mol/contact-residue — check training data coverage."
        )
    if not (_BETA_MIN <= beta <= _BETA_MAX):
        raise ValueError(
            f"Calibrated β={beta:.3f} is outside valid range [{_BETA_MIN}, {_BETA_MAX}] "
            "— β > 0.5 means AD4 dominates over Vina, contradicting the Vina-primary design. "
            "Check training data or use default calibration.json."
        )

    gamma = float(cal.get("gamma", 0.0))
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(
            f"Calibrated γ={gamma:.3f} is outside valid range [0.0, 1.0] — "
            "γ is the non-contact residue entropy fraction. Check calibration file."
        )

    ensemble_ad4_weight = float(cal.get("ensemble_ad4_weight", 0.0))
    if not (0.0 <= ensemble_ad4_weight <= 1.0):
        raise ValueError(
            f"ensemble_ad4_weight={ensemble_ad4_weight:.3f} outside [0.0, 1.0]. "
            "Check calibration file."
        )

    _log.debug(
        "Loaded calibration: alpha=%.3f beta=%.3f gamma=%.3f ensemble_ad4_weight=%.3f from %s",
        alpha, beta, gamma, ensemble_ad4_weight, path,
    )
    return cal


def write_calibration(
    path: Path,
    alpha: float,
    beta: float,
    **kwargs: float | int | str,
) -> None:
    """Write calibration parameters to a JSON file.

    Always sets 'calibrated_at' to the current UTC time in ISO 8601 format.
    Creates parent directories as needed.

    Args:
        path: Destination path for the calibration JSON file.
        alpha: Burial correction coefficient (kcal/mol/contact-residue).
        beta: AD4 blending weight (dimensionless, [0.0, 0.5]).
        **kwargs: Additional fields to include (e.g., n_complexes,
            pearson_r, rmse_kcal_mol, training_csv).
    """
    payload = {
        "alpha": alpha,
        "beta": beta,
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
        **kwargs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    _log.info("Wrote calibration to %s (alpha=%.3f, beta=%.3f)", path, alpha, beta)


def _parse_heavy_atoms_by_residue(pdb_path: Path) -> list[np.ndarray]:
    """Parse heavy atom coordinates from PDB grouped by residue.

    Args:
        pdb_path: Path to PDB file.

    Returns:
        List of (n_atoms, 3) float64 arrays, one per unique (chain, resseq, resname).
    """
    residues: dict[tuple[str, int, str], list[tuple[float, float, float]]] = {}
    with pdb_path.open() as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            if not atom_name or atom_name[0] in ("H", "D"):
                continue
            try:
                chain = line[21]
                res_seq = int(line[22:26])
                res_name = line[17:20].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            key = (chain, res_seq, res_name)
            residues.setdefault(key, []).append((x, y, z))
    return [np.array(coords, dtype=np.float64) for coords in residues.values()]


def _parse_heavy_atom_coords(pdb_path: Path) -> np.ndarray:
    """Parse all heavy atom coordinates from PDB as (N, 3) float64 array.

    Args:
        pdb_path: Path to PDB file.

    Returns:
        (N, 3) array of XYZ coordinates; empty (0, 3) if no atoms found.
    """
    coords: list[tuple[float, float, float]] = []
    with pdb_path.open() as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            if not atom_name or atom_name[0] in ("H", "D"):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            coords.append((x, y, z))
    return np.array(coords, dtype=np.float64) if coords else np.empty((0, 3), dtype=np.float64)


def load_receptor_heavy_atom_coords(receptor_pdb: Path) -> np.ndarray:
    """Load receptor heavy atom coordinates for contact/clash calculations.

    Intended to be called ONCE per docking run and reused across all poses
    for efficiency. Uses _parse_heavy_atom_coords internally.

    Args:
        receptor_pdb: Path to the receptor PDB file.

    Returns:
        (N, 3) float64 array of receptor heavy atom XYZ coordinates.
    """
    coords = _parse_heavy_atom_coords(receptor_pdb)
    _log.debug(
        "Loaded %d receptor heavy atoms from %s", len(coords), receptor_pdb
    )
    return coords


def count_contact_residues(
    pose_pdb: Path,
    receptor_coords: np.ndarray,
    cutoff: float = CONTACT_DIST_ANG,
) -> int:
    """Count peptide residues with at least one heavy atom within cutoff of receptor.

    A residue is counted as a contact residue if any of its heavy atoms are
    within ``cutoff`` Angstroms of any receptor heavy atom. This gives a
    physically meaningful measure of binding interface size, and is used
    instead of full sequence length in the burial correction to avoid
    over-penalizing peptides where most residues are disordered / not
    contacting the protein.

    Args:
        pose_pdb: Path to the peptide pose PDB file.
        receptor_coords: (N, 3) array of receptor heavy atom coordinates,
            pre-loaded via load_receptor_heavy_atom_coords() for efficiency.
        cutoff: Distance cutoff in Angstroms. Default CONTACT_DIST_ANG (4.5 Å).

    Returns:
        Number of residues with at least one heavy atom within cutoff.
        Returns 0 if receptor_coords is empty or no atoms are parsed.
    """
    if len(receptor_coords) == 0:
        return 0
    per_residue = _parse_heavy_atoms_by_residue(pose_pdb)
    if not per_residue:
        return 0
    n_contact = 0
    for res_coords in per_residue:
        if len(res_coords) == 0:
            continue
        # Broadcast: (n_res_atoms, 1, 3) − (1, n_rec_atoms, 3) → (n_res, n_rec, 3)
        diffs = res_coords[:, np.newaxis, :] - receptor_coords[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=-1)  # (n_res, n_rec)
        if sq_dists.min() <= cutoff ** 2:
            n_contact += 1
    return n_contact


def check_intermolecular_clash(
    pose_pdb: Path,
    receptor_coords: np.ndarray,
    clash_cutoff: float = 1.5,
) -> bool:
    """Check if any peptide heavy atom is within clash_cutoff of any receptor heavy atom.

    A clash at < 1.5 Å indicates the pose is geometrically inside the receptor
    (peptide diffused through the receptor surface). Such poses have catastrophically
    positive Vina scores and should be excluded from ranking.

    Args:
        pose_pdb: Path to peptide pose PDB file.
        receptor_coords: (N, 3) array of receptor heavy atom coordinates.
        clash_cutoff: Minimum acceptable distance in Angstroms. Default 1.5 Å.

    Returns:
        True if any peptide heavy atom is within clash_cutoff of any receptor atom.
    """
    if len(receptor_coords) == 0:
        return False
    peptide_coords = _parse_heavy_atom_coords(pose_pdb)
    if len(peptide_coords) == 0:
        return False
    diffs = peptide_coords[:, np.newaxis, :] - receptor_coords[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=-1)
    return bool(sq_dists.min() < clash_cutoff ** 2)


def apply_hybrid_score(
    pose: ScoredPose,
    *,
    alpha: float,
    beta: float,
    n_residues: int,
    n_contact_residues: int | None = None,
    gamma: float = 0.0,
) -> None:
    """Apply the hybrid score formula to a ScoredPose in place.

    Sets ``pose.entropy_correction = alpha * n_eff`` and
    ``pose.hybrid_score = vina + effective_beta*(ad4 - vina) + alpha*n_eff``

    n_eff accounts for differential entropy by residue contact state:
      n_eff = n_contact + gamma * n_non_contact
    where n_non_contact = max(0, n_residues - n_contact).

    gamma=0.0 (default): only contact residues pay the entropy penalty.
    gamma=0.2: non-contact residues pay 20% of the per-residue penalty.
    This reflects that tethered non-contact residues lose some translational
    freedom even when not directly interfacing the receptor.

    effective_beta is 0 when ``pose.is_ad4_anomaly`` is True (AD4 > 0
    indicates a corrupt/clashed pose; bypassing it prevents the anomalous
    positive AD4 from worsening the hybrid score).

    Args:
        pose: ScoredPose with vina_score and ad4_score already set.
        alpha: Backbone entropy coefficient (kcal/mol/contact-residue).
        beta: AD4 blending weight (dimensionless).
        n_residues: Full peptide length; used to compute non-contact count.
        n_contact_residues: Number of residues in contact with receptor (≤ n_residues).
            When None, falls back to n_residues (n_non_contact = 0, gamma has no effect).
        gamma: Fraction of alpha applied to non-contact residues [0.0, 1.0].
            Loaded from calibration.json; default 0.0 (contact-only, legacy mode).

    Raises:
        RuntimeError: If pose.vina_score is None.
    """
    if pose.vina_score is None:
        raise RuntimeError(
            f"Pose {pose.pose_idx}: vina_score is None — Vina scoring must run before apply_hybrid_score"
        )

    n_contact = n_contact_residues if n_contact_residues is not None else n_residues
    n_non_contact = max(0, n_residues - n_contact)
    n_eff = n_contact + gamma * n_non_contact
    pose.entropy_correction = alpha * n_eff

    # AD4 anomaly bypass: when AD4 score is physically meaningless (> 0),
    # fall back to Vina-only weighting rather than letting a large positive
    # AD4 contaminate the hybrid score.
    effective_beta = 0.0 if pose.is_ad4_anomaly else beta

    if pose.ad4_score is not None:
        pose.hybrid_score = (
            pose.vina_score
            + effective_beta * (pose.ad4_score - pose.vina_score)
            + pose.entropy_correction
        )
        _log.debug(
            "Pose %d: vina=%.3f ad4=%.3f ec=%.3f hybrid=%.3f "
            "(beta_eff=%.3f%s n_contact=%d n_noc=%d γ=%.2f)",
            pose.pose_idx,
            pose.vina_score,
            pose.ad4_score,
            pose.entropy_correction,
            pose.hybrid_score,
            effective_beta,
            " AD4-bypassed" if pose.is_ad4_anomaly else "",
            n_contact,
            n_non_contact,
            gamma,
        )
    else:
        pose.hybrid_score = pose.vina_score + pose.entropy_correction
        _log.debug(
            "Pose %d: vina=%.3f ad4=None ec=%.3f hybrid=%.3f "
            "(vina-only n_contact=%d n_noc=%d γ=%.2f)",
            pose.pose_idx,
            pose.vina_score,
            pose.entropy_correction,
            pose.hybrid_score,
            n_contact,
            n_non_contact,
            gamma,
        )


def fit_calibration(
    vina_scores: list[float],
    ad4_scores: list[float],
    n_residues_list: list[int],
    experimental_pkd: list[float],
    n_contact_residues_list: list[int] | None = None,
    gamma: float = 0.2,
) -> dict:
    """Fit calibration parameters (alpha, beta) using L-BFGS-B minimization.

    For n > 2 training complexes, directly optimises Pearson r (maximises
    correlation between predicted and experimental ΔG) because Pearson r is
    the primary benchmark metric. For n ≤ 2, falls back to SSE (Pearson r is
    trivially 1.0 for two points regardless of objective).

    n_eff per training complex is computed with the same gamma formula used at
    inference time:
        n_eff = n_contact + gamma * (n_residues - n_contact)

    When ``n_contact_residues_list`` is provided, the entropy correction uses
    contact counts + gamma*non-contact. The alpha upper bound is 2.0 to
    accommodate contact-based mode where fewer effective residues require a
    larger per-residue coefficient.

    Bounds enforced by scipy L-BFGS-B:
        alpha ∈ [0.1, 2.0]
        beta  ∈ [0.0, 0.5]

    Starting point: x0 = [0.65, 0.22] (empirical defaults).

    Args:
        vina_scores: List of Vina --score_only values in kcal/mol.
        ad4_scores: List of AutoDock4 scoring values in kcal/mol.
        n_residues_list: List of full peptide lengths.
        experimental_pkd: List of experimental pKd values.
        n_contact_residues_list: Optional list of contact residue counts per complex.
            When provided, used with gamma to compute n_eff per complex.
        gamma: Non-contact residue entropy fraction [0.0, 1.0]. Default 0.2.
            Stored in returned dict and written to calibration.json.

    Returns:
        Dictionary with keys: 'alpha', 'beta', 'pearson_r', 'rmse_kcal_mol',
        'entropy_mode' ('contact' or 'residue'), 'gamma'.

    Raises:
        ValueError: If input lists have different lengths.
    """
    n = len(vina_scores)
    if not (n == len(ad4_scores) == len(n_residues_list) == len(experimental_pkd)):
        raise ValueError(
            "All input lists must have the same length; got lengths: "
            f"vina={len(vina_scores)}, ad4={len(ad4_scores)}, "
            f"n_res={len(n_residues_list)}, pkd={len(experimental_pkd)}"
        )
    if n_contact_residues_list is not None and len(n_contact_residues_list) != n:
        raise ValueError(
            f"n_contact_residues_list length {len(n_contact_residues_list)} "
            f"must match other inputs (n={n})"
        )

    if n_contact_residues_list is not None:
        n_eff_list = [
            nc + gamma * max(0, nr - nc)
            for nc, nr in zip(n_contact_residues_list, n_residues_list)
        ]
        entropy_mode = "contact"
    else:
        n_eff_list = list(n_residues_list)
        entropy_mode = "residue"

    delta_g = [-_RT * _LN10 * pkd for pkd in experimental_pkd]

    def _hybrids(alpha: float, beta: float) -> list[float]:
        return [
            v + beta * (a - v) + alpha * nr
            for v, a, nr in zip(vina_scores, ad4_scores, n_eff_list)
        ]

    def objective(params: np.ndarray) -> float:
        alpha, beta = float(params[0]), float(params[1])
        hybrids = _hybrids(alpha, beta)
        if n > 2:
            # Directly maximise Pearson r (minimise negative r) for n > 2.
            # With only 2 points, r is always ±1 regardless of objective.
            r, _ = pearsonr(hybrids, delta_g)
            return -float(r)
        residuals = [h - dg for h, dg in zip(hybrids, delta_g)]
        return float(sum(r**2 for r in residuals))

    x0 = np.array([0.65, 0.22])
    bounds = [(_ALPHA_MIN, _ALPHA_MAX), (_BETA_MIN, _BETA_MAX)]
    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        _log.warning(
            "L-BFGS-B optimization did not converge: %s. "
            "Proceeding with best-found parameters — verify calibration manually.",
            result.message,
        )
    alpha, beta = float(result.x[0]), float(result.x[1])

    hybrids = _hybrids(alpha, beta)
    if len(hybrids) > 1:
        r, _ = pearsonr(hybrids, delta_g)
        pearson_r = float(r)
    else:
        pearson_r = float("nan")

    rmse = float(np.sqrt(np.mean([(h - d) ** 2 for h, d in zip(hybrids, delta_g)])))

    _log.info(
        "fit_calibration (%s mode, γ=%.2f): alpha=%.4f beta=%.4f r=%.3f rmse=%.3f",
        entropy_mode,
        gamma,
        alpha,
        beta,
        pearson_r,
        rmse,
    )
    return {
        "alpha": alpha,
        "beta": beta,
        "pearson_r": pearson_r,
        "rmse_kcal_mol": rmse,
        "entropy_mode": entropy_mode,
        "gamma": gamma,
    }


def apply_ensemble_hybrid_scores(
    poses: list[ScoredPose],
    *,
    alpha: float,
    n_residues: int,
    ad4_blend_weight: float = 0.3,
    gamma: float = 0.0,
) -> None:
    """Apply hybrid scores using within-ensemble z-score normalization of AD4.

    Standard per-pose scoring (apply_hybrid_score) blends Vina and AD4 using
    absolute-scale calibration (beta). When beta calibrates to 0 — which
    happens when crystal-pose Vina scores already overshoot experimental ΔG
    — AD4 contributes nothing even though it carries real electrostatic signal.

    This function instead normalises Vina and AD4 scores to z-scores within
    the current pose ensemble, blends the z-scores with ad4_blend_weight, then
    back-projects to kcal/mol using the Vina distribution parameters. The
    physical motivation: AD4 (Gasteiger charges) reliably ranks poses by
    electrostatic complementarity *relative to each other*, even when its
    absolute ΔG is poorly calibrated on peptides. Z-score normalisation
    separates the ranking signal from the absolute-scale calibration problem.

    Only non-anomalous AD4 scores (is_ad4_anomaly=False) enter the AD4
    distribution. Poses with anomalous or missing AD4 scores fall back to
    Vina-only z-score for that pose.

    Modifies poses in-place (sets entropy_correction and hybrid_score).

    Args:
        poses: Scored poses; each must have vina_score set.
        alpha: Backbone entropy coefficient (kcal/mol/contact-residue).
        n_residues: Full peptide length.
        ad4_blend_weight: Fraction of the z-score blend assigned to AD4 [0, 1].
            Default 0.3 (30% AD4, 70% Vina). Set to 0 to disable AD4.
        gamma: Non-contact residue entropy fraction [0, 1]. Default 0.0.

    Raises:
        RuntimeError: If no poses have valid Vina scores.
    """
    vina_vals = [p.vina_score for p in poses if p.vina_score is not None]
    if not vina_vals:
        raise RuntimeError(
            "apply_ensemble_hybrid_scores: no poses with valid Vina scores"
        )

    v_mean = float(np.mean(vina_vals))
    v_std = float(np.std(vina_vals)) if len(vina_vals) > 1 else 1.0
    if v_std == 0.0:
        v_std = 1.0

    ad4_vals = [
        p.ad4_score
        for p in poses
        if p.ad4_score is not None and not p.is_ad4_anomaly
    ]
    has_ad4 = len(ad4_vals) >= 2 and ad4_blend_weight > 0.0
    a_mean = a_std = 0.0
    if has_ad4:
        a_mean = float(np.mean(ad4_vals))
        a_std = float(np.std(ad4_vals)) if len(ad4_vals) > 1 else 1.0
        if a_std == 0.0:
            a_std = 1.0
        _log.info(
            "Ensemble scoring: v_mean=%.3f v_std=%.3f a_mean=%.3f a_std=%.3f "
            "ad4_weight=%.2f n_valid_ad4=%d",
            v_mean, v_std, a_mean, a_std, ad4_blend_weight, len(ad4_vals),
        )
    else:
        _log.info(
            "Ensemble scoring: Vina-only (ad4_blend_weight=%.2f n_valid_ad4=%d)",
            ad4_blend_weight, len(ad4_vals),
        )

    for pose in poses:
        if pose.vina_score is None:
            continue

        n_contact = pose.n_contact_residues if pose.n_contact_residues is not None else n_residues
        n_non_contact = max(0, n_residues - n_contact)
        n_eff = n_contact + gamma * n_non_contact
        pose.entropy_correction = alpha * n_eff

        z_vina = (pose.vina_score - v_mean) / v_std

        if has_ad4 and pose.ad4_score is not None and not pose.is_ad4_anomaly:
            z_ad4 = (pose.ad4_score - a_mean) / a_std
            blended_z = (1.0 - ad4_blend_weight) * z_vina + ad4_blend_weight * z_ad4
        else:
            blended_z = z_vina

        pose.hybrid_score = v_mean + v_std * blended_z + pose.entropy_correction
        _log.debug(
            "Pose %d: vina=%.3f hybrid=%.3f (z_v=%.3f%s ec=%.3f)",
            pose.pose_idx,
            pose.vina_score,
            pose.hybrid_score,
            z_vina,
            f" z_a={z_ad4:.3f}" if (has_ad4 and pose.ad4_score is not None and not pose.is_ad4_anomaly) else "",
            pose.entropy_correction,
        )
