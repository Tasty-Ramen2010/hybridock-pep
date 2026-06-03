"""Backbone entropy (contact-burial) correction and hybrid score calibration (SCORE-03).

Implements the hybrid score formula (see docs/specs/README.md for D-XX ID legend):
    hybrid = vina + beta*(ad4 - vina) + alpha*n_eff_residues

where ``alpha * n_eff_residues`` is the backbone contact-burial correction term
(called "entropy" for historical reasons — it is a contact-count penalty, not a true
thermodynamic entropy calculation), ``n_eff_residues`` is either the full peptide length
or the contact-residue count (residues with at least one heavy atom within
``CONTACT_DIST_ANG`` (4.5 Å) of the receptor), and ``beta`` controls the blending
weight of AD4 relative to Vina.

NOTE ON TERMINOLOGY
    The "entropy" label in this module refers to the contact-count burial correction
    (``alpha × n_contact``), which approximates the entropic cost of peptide burial
    at the protein–peptide interface.  It is **not** a true entropy calculation —
    no phase-space integration or partition function is evaluated.  The name is a
    shorthand adopted from implicit-solvent literature (Lazaridis & Karplus, 1999;
    Street & Mayo, 1998) where contact number serves as a proxy for solvation
    entropy change upon burial.  Calibrated ``alpha`` absorbs both the magnitude
    and sign of this effect from experimental pKd data via L-BFGS-B optimization.

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

# Schema v2 (multivariate ridge) bounds.  Wider than legacy because the
# ridge fit has direct linear weights instead of the constrained α/β
# formulation.  See docs/calibration_notes.md "Production-pose v2" section.
# Strict v2 ridge bounds — a single global fit must respect Vina's sign
# convention (more-negative = better binding). Per-family fits (schema v3)
# can validly invert weights inside a narrow family where the per-family
# intercept dominates and within-family slopes are noisy, so v3 uses
# _W_VINA_MIN_LOOSE for its slope bounds.
_W_VINA_MIN, _W_VINA_MAX = 0.0, 3.0
_W_VINA_MIN_LOOSE = -2.0
_W_AD4_MIN,  _W_AD4_MAX  = -1.0, 3.0
_W_CONTACT_MIN, _W_CONTACT_MAX = -3.0, 3.0
_INTERCEPT_MIN, _INTERCEPT_MAX = -30.0, 30.0

# Contact distance threshold (Fix A — unified across inference and calibration).
# ALL code that counts contact residues (driver.py, score_crystal_poses.py,
# score_calibration_set.py) must import and use this constant so that α is
# calibrated on the same contact counts that inference uses at prediction time.
# Changing this value requires re-running Tier 1.3 calibration.
CONTACT_DIST_ANG: float = 4.5  # Å heavy-atom distance for "in contact" classification


def calibration_mode(cal: dict) -> str:
    """Return 'per_family', 'ridge', or 'legacy' based on calibration schema.

    Detection rule (in priority order):
      1. ``cal["model_type"] == "per_family_ridge"`` → per_family
      2. ``cal["schema_version"] >= 3`` and ``families`` present → per_family
      3. ``cal["model_type"] == "ridge"`` → ridge
      4. ``cal["schema_version"] >= 2`` and ``w_vina`` present → ridge
      5. otherwise → legacy

    Per-family (schema v3) carries a ``families`` dict mapping cluster IDs to
    per-family ridge fits, plus a ``fallback`` ridge for out-of-distribution
    receptors. Ridge (schema v2) carries flat per-feature weights. Legacy v1
    carries ``alpha`` + ``beta``.

    Args:
        cal: A calibration dict returned by ``load_calibration``.

    Returns:
        "per_family", "ridge", or "legacy".
    """
    model_type = cal.get("model_type")
    if model_type == "per_family_ridge":
        return "per_family"
    if int(cal.get("schema_version", 1)) >= 3 and isinstance(cal.get("families"), dict):
        return "per_family"
    if model_type == "ridge":
        return "ridge"
    if model_type == "legacy" or model_type == "single_alpha":
        return "legacy"
    if int(cal.get("schema_version", 1)) >= 2 and "w_vina" in cal:
        return "ridge"
    return "legacy"


def load_calibration(path: Path) -> dict:
    """Load and validate calibration parameters from a JSON file.

    Supports two on-disk schemas, dispatched by ``calibration_mode``:

    * **Legacy (v1)** — single-α model: requires ``alpha`` ∈ [0.1, 2.0]
      and ``beta`` ∈ [0.0, 0.5]; ``gamma`` optional in [0.0, 1.0];
      ``ensemble_ad4_weight`` optional in [0.0, 1.0].  Production
      formula: ``hybrid = vina + beta*(ad4 - vina) + alpha*n_eff``.
    * **Ridge (v2)** — explicit per-feature weights: requires
      ``w_vina`` ∈ [0, 3], ``w_ad4`` ∈ [-1, 3], ``w_contact`` ∈ [-3, 3],
      ``intercept`` ∈ [-30, 30].  Production formula:
      ``hybrid = w_vina*vina + w_ad4*ad4 + w_contact*n_contact + intercept``.
      The ``w_contact`` sign is intuitive: negative means more contacts
      → more binding (hydrophobic burial), positive means the legacy
      entropy-penalty direction.

    Args:
        path: Path to the calibration JSON file.

    Returns:
        Dictionary containing calibration data.  Use
        ``calibration_mode(dict)`` to dispatch on schema.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If schema-specific fields are missing or out of range.
    """
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    with path.open() as fh:
        cal = json.load(fh)

    mode = calibration_mode(cal)

    if mode == "per_family":
        # Validate that the schema-v3 file has both families and fallback,
        # and that each family carries the same ridge keys we validate for v2.
        families = cal.get("families")
        fallback = cal.get("fallback")
        if not isinstance(families, dict) or not families:
            raise ValueError(
                f"Per-family calibration {path} has no non-empty 'families' dict."
            )
        if not isinstance(fallback, dict):
            raise ValueError(
                f"Per-family calibration {path} requires a 'fallback' ridge dict."
            )
        _validate_ridge(fallback, path, loose=True)
        for fam_id, fit in families.items():
            _validate_ridge(fit, path, loose=True)
        _log.debug(
            "Loaded per-family calibration: %d families + fallback from %s",
            len(families), path,
        )
        return cal

    if mode == "ridge":
        _validate_ridge(cal, path)
        _log.debug(
            "Loaded ridge calibration: w_vina=%.3f w_ad4=%.3f w_contact=%.3f "
            "intercept=%.3f from %s",
            cal["w_vina"], cal["w_ad4"], cal["w_contact"], cal["intercept"], path,
        )
        return cal

    # Legacy single-α schema.
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


def _validate_ridge(cal: dict, path: Path, *, loose: bool = False) -> None:
    """Validate the ridge / v2 calibration keys and ranges.  Raises ValueError on bad input.

    Args:
        cal: Ridge fit dict.
        path: Source file path (for error messages).
        loose: If True, allow negative w_vina (per-family ridges in schema v3
            can validly have inverted slopes when the cluster intercept does
            most of the work and the within-family Vina trend is weak/noisy).
    """
    required = ("w_vina", "w_ad4", "w_contact", "intercept")
    missing = [k for k in required if k not in cal]
    if missing:
        raise ValueError(
            f"Ridge calibration {path} is missing required keys: {missing}.  "
            "Re-run calibrate_alpha.py --mode ridge to regenerate."
        )
    w_vina_lo = _W_VINA_MIN_LOOSE if loose else _W_VINA_MIN
    pairs = [
        ("w_vina",    cal["w_vina"],    w_vina_lo,      _W_VINA_MAX),
        ("w_ad4",     cal["w_ad4"],     _W_AD4_MIN,     _W_AD4_MAX),
        ("w_contact", cal["w_contact"], _W_CONTACT_MIN, _W_CONTACT_MAX),
        ("intercept", cal["intercept"], _INTERCEPT_MIN, _INTERCEPT_MAX),
    ]
    # Optional per-residue entropy weights (schema v2 extension): only
    # validated when present.  Range matches w_contact (-3..3 kcal/mol per
    # unit of the corresponding entropy proxy).
    for opt_key in ("w_s_sc", "w_s_bb", "w_s_ss_weighted"):
        if opt_key in cal:
            pairs.append((opt_key, cal[opt_key], _W_CONTACT_MIN, _W_CONTACT_MAX))
    for name, val, lo, hi in pairs:
        if not (lo <= float(val) <= hi):
            raise ValueError(
                f"Ridge calibration {path}: {name}={val:.3f} is outside valid "
                f"range [{lo}, {hi}].  Re-run calibration or update the bounds "
                "in scoring/entropy.py if this is intentional."
            )


def write_calibration(
    path: Path,
    alpha: float | None = None,
    beta: float | None = None,
    **kwargs: float | int | str | bool,
) -> None:
    """Write calibration parameters to a JSON file.

    Supports both schemas.  Pass ``alpha`` and ``beta`` for a legacy
    single-α calibration.  Pass ``schema_version=2``, ``model_type="ridge"``,
    plus ``w_vina``/``w_ad4``/``w_contact``/``intercept`` via kwargs for a
    ridge calibration; in that case ``alpha`` and ``beta`` may be omitted.

    Always sets 'calibrated_at' to the current UTC time in ISO 8601 format.
    Creates parent directories as needed.

    Args:
        path: Destination path for the calibration JSON file.
        alpha: Burial correction coefficient (kcal/mol/contact-residue);
            required for legacy schema.
        beta: AD4 blending weight (dimensionless, [0.0, 0.5]);
            required for legacy schema.
        **kwargs: Additional fields.  For ridge schema, MUST include
            ``schema_version``, ``model_type``, ``w_vina``, ``w_ad4``,
            ``w_contact``, ``intercept``.
    """
    payload: dict = {"calibrated_at": datetime.now(timezone.utc).isoformat()}
    if alpha is not None:
        payload["alpha"] = alpha
    if beta is not None:
        payload["beta"] = beta
    payload.update(kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    if calibration_mode(payload) == "ridge":
        _log.info(
            "Wrote ridge calibration to %s (w_vina=%.3f w_ad4=%.3f w_contact=%.3f intercept=%.3f)",
            path, payload["w_vina"], payload["w_ad4"], payload["w_contact"], payload["intercept"],
        )
    else:
        _log.info(
            "Wrote calibration to %s (alpha=%.3f, beta=%.3f)",
            path, payload.get("alpha", float("nan")), payload.get("beta", float("nan")),
        )


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

    **Calibration consistency:** The default cutoff is ``CONTACT_DIST_ANG`` (4.5 Å).
    All calibration scripts must use the same constant so that α is fitted on contact
    counts computed with the same cutoff used at inference time. Do not pass a
    different cutoff without re-running Tier 1.3 calibration.

    Args:
        pose_pdb: Path to the peptide pose PDB file.
        receptor_coords: (N, 3) array of receptor heavy atom coordinates,
            pre-loaded via load_receptor_heavy_atom_coords() for efficiency.
        cutoff: Distance cutoff in Angstroms. Default ``CONTACT_DIST_ANG`` (4.5 Å).

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
        alpha: Burial correction coefficient (kcal/mol/contact-residue).
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


def apply_hybrid_score_ridge(
    pose: ScoredPose,
    *,
    w_vina: float,
    w_ad4: float,
    w_contact: float,
    intercept: float,
    n_contact_residues: int,
    w_s_sc: float = 0.0,
    w_s_bb: float = 0.0,
    w_s_ss_weighted: float = 0.0,
) -> None:
    """Apply the schema-v2 multivariate ridge hybrid-score formula.

    ``hybrid = w_vina * vina
             + w_ad4 * ad4
             + w_contact * n_contact
             + w_s_sc * s_sc_sum
             + w_s_bb * s_bb_sum
             + w_s_ss_weighted * s_ss_weighted
             + intercept``

    Per-residue entropy weights default to 0.0 so existing v2 calibrations
    that only set ``w_vina`` / ``w_ad4`` / ``w_contact`` / ``intercept``
    behave identically.  When ``w_s_*`` are non-zero, the corresponding
    ``pose.s_*`` field must be populated (computed in driver.py Stage 2d-pre
    via ``per_residue_entropy.compute_entropy_sums``).

    Sign convention: negative weights on entropy features mean "more
    entropy proxy → more binding," matching the PepSet-6 production-pose
    ridge result.  Positive weights match the legacy entropy-penalty
    direction.

    AD4 anomaly bypass: when ``pose.is_ad4_anomaly`` is True or
    ``pose.ad4_score`` is None, the AD4 term is dropped (effective w_ad4 = 0).

    Args:
        pose: ScoredPose with vina_score set; ad4_score / s_*_sum optional.
        w_vina: Vina coefficient.
        w_ad4: AD4 coefficient.
        w_contact: Contact-count coefficient.
        intercept: Constant offset, kcal/mol.
        n_contact_residues: Peptide residues in contact with receptor (≥ 0).
        w_s_sc: Side-chain Doig-Sternberg entropy sum coefficient.  Default 0.
        w_s_bb: Backbone entropy sum coefficient.  Default 0.
        w_s_ss_weighted: SS-weighted (s_sc + s_bb) sum coefficient.
            Default 0.  See scoring/per_residue_entropy.py for the formula.

    Raises:
        RuntimeError: If ``pose.vina_score`` is None, or if a non-zero
            ``w_s_*`` weight is set but the corresponding pose field is None.
    """
    if pose.vina_score is None:
        raise RuntimeError(
            f"Pose {pose.pose_idx}: vina_score is None — Vina must run before apply_hybrid_score_ridge"
        )
    use_ad4 = pose.ad4_score is not None and not pose.is_ad4_anomaly
    ad4_term = w_ad4 * pose.ad4_score if use_ad4 else 0.0

    # Per-residue entropy terms (zero by default; verified populated when used)
    sc_term = 0.0
    bb_term = 0.0
    ss_term = 0.0
    if w_s_sc != 0.0:
        if pose.s_sc_sum is None:
            raise RuntimeError(
                f"Pose {pose.pose_idx}: w_s_sc set but pose.s_sc_sum is None — "
                "compute_entropy_sums must run before apply_hybrid_score_ridge"
            )
        sc_term = w_s_sc * float(pose.s_sc_sum)
    if w_s_bb != 0.0:
        if pose.s_bb_sum is None:
            raise RuntimeError(
                f"Pose {pose.pose_idx}: w_s_bb set but pose.s_bb_sum is None"
            )
        bb_term = w_s_bb * float(pose.s_bb_sum)
    if w_s_ss_weighted != 0.0:
        if pose.s_ss_weighted is None:
            raise RuntimeError(
                f"Pose {pose.pose_idx}: w_s_ss_weighted set but pose.s_ss_weighted is None"
            )
        ss_term = w_s_ss_weighted * float(pose.s_ss_weighted)

    contact_term = w_contact * float(n_contact_residues)
    # Single number capturing the full non-Vina/non-AD4 entropy contribution.
    pose.entropy_correction = contact_term + sc_term + bb_term + ss_term
    pose.hybrid_score = (
        w_vina * float(pose.vina_score)
        + ad4_term
        + pose.entropy_correction
        + intercept
    )
    _log.debug(
        "Pose %d (ridge): vina=%.3f ad4=%s nC=%d s_sc=%s s_bb=%s s_ss=%s "
        "hybrid=%.3f (w_v=%.3f w_a=%.3f w_c=%.3f w_sc=%.3f w_bb=%.3f w_ss=%.3f intc=%.3f)",
        pose.pose_idx,
        float(pose.vina_score),
        f"{float(pose.ad4_score):.3f}" if use_ad4 else "skipped",
        n_contact_residues,
        f"{pose.s_sc_sum:.2f}" if pose.s_sc_sum is not None else "—",
        f"{pose.s_bb_sum:.2f}" if pose.s_bb_sum is not None else "—",
        f"{pose.s_ss_weighted:.2f}" if pose.s_ss_weighted is not None else "—",
        pose.hybrid_score,
        w_vina, w_ad4, w_contact, w_s_sc, w_s_bb, w_s_ss_weighted, intercept,
    )


def apply_calibration(
    pose: ScoredPose,
    cal: dict,
    n_residues: int,
    n_contact_residues: int | None = None,
    ridge_override: dict | None = None,
) -> None:
    """Dispatch to the correct hybrid-score formula based on calibration schema.

    Reads ``calibration_mode(cal)`` and applies one of three paths:

    * ``legacy`` (v1, single-α): ``apply_hybrid_score``.
    * ``ridge``  (v2, multivariate): ``apply_hybrid_score_ridge``.
    * ``per_family`` (v3): the caller is expected to have already routed the
      receptor to a per-family ridge via ``apply_per_family_calibration``
      and to pass the resolved ridge fit via ``ridge_override``. If
      ``ridge_override`` is None, the fallback ridge is used. This
      double-dispatch avoids reloading the receptor PDB per pose.

    Args:
        pose: ScoredPose with vina_score populated.
        cal: Calibration dict from ``load_calibration``.
        n_residues: Full peptide length (used by legacy gamma path).
        n_contact_residues: Contact residue count. If None, the legacy path
            falls back to ``n_residues``; the ridge path treats missing as 0.
        ridge_override: When set, treated as the per-feature ridge to apply
            (must have ``w_vina``, ``w_ad4`` etc. keys). Used by the per-family
            dispatcher to inject the routed family's ridge.
    """
    mode = calibration_mode(cal)
    if mode == "per_family":
        # Caller should have routed; honor override, else fallback ridge.
        chosen = ridge_override if ridge_override is not None else cal.get("fallback")
        if chosen is None:
            raise ValueError(
                "per-family calibration has no 'fallback' ridge and no ridge_override"
            )
        apply_hybrid_score_ridge(
            pose,
            w_vina=float(chosen["w_vina"]),
            w_ad4=float(chosen.get("w_ad4", 0.0)),
            w_contact=float(chosen["w_contact"]),
            intercept=float(chosen["intercept"]),
            n_contact_residues=int(n_contact_residues or 0),
            w_s_sc=float(chosen.get("w_s_sc", 0.0)),
            w_s_bb=float(chosen.get("w_s_bb", 0.0)),
            w_s_ss_weighted=float(chosen.get("w_s_ss_weighted", 0.0)),
        )
        return
    if mode == "ridge":
        apply_hybrid_score_ridge(
            pose,
            w_vina=float(cal["w_vina"]),
            w_ad4=float(cal["w_ad4"]),
            w_contact=float(cal["w_contact"]),
            intercept=float(cal["intercept"]),
            n_contact_residues=int(n_contact_residues or 0),
            w_s_sc=float(cal.get("w_s_sc", 0.0)),
            w_s_bb=float(cal.get("w_s_bb", 0.0)),
            w_s_ss_weighted=float(cal.get("w_s_ss_weighted", 0.0)),
        )
        return
    apply_hybrid_score(
        pose,
        alpha=float(cal["alpha"]),
        beta=float(cal["beta"]),
        n_residues=n_residues,
        n_contact_residues=n_contact_residues,
        gamma=float(cal.get("gamma", 0.0)),
    )


def fit_calibration_ridge(
    vina_scores: list[float],
    ad4_scores: list[float],
    n_contact_residues_list: list[int],
    experimental_pkd: list[float],
    *,
    ridge_alpha: float = 0.1,
    positive: bool = True,
) -> dict:
    """Fit the schema-v2 ridge model ``ΔG = w_vina·V + w_ad4·A + w_contact·N + c``.

    Convention: features are (vina, ad4, ``-n_contact``) so the fitted
    third weight is positive when "more contacts → more binding".  The
    returned ``w_contact`` is the **negated** value (intuitive sign:
    negative = binding direction; matches ``apply_hybrid_score_ridge``).

    Reports both in-sample Pearson r and leave-one-out CV r when n ≥ 3.

    Args:
        vina_scores: Per-complex Vina aggregate (kcal/mol).
        ad4_scores: Per-complex AD4 aggregate (kcal/mol).
        n_contact_residues_list: Per-complex contact count.
        experimental_pkd: Per-complex experimental pKd.
        ridge_alpha: Ridge regularisation strength.  Default 0.1.
        positive: If True, weights are constrained ≥ 0 (in the
            (vina, ad4, -n_contact) basis), which means w_contact ≤ 0
            after negation.

    Returns:
        Dict suitable for ``write_calibration`` and round-trip through
        ``load_calibration`` (schema v2 / ridge).
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import LeaveOneOut
    except ImportError as exc:
        raise RuntimeError(
            "fit_calibration_ridge requires scikit-learn.  "
            "Install in score-env: pip install scikit-learn"
        ) from exc

    n = len(vina_scores)
    lengths = (n, len(ad4_scores), len(n_contact_residues_list), len(experimental_pkd))
    if len(set(lengths)) != 1:
        raise ValueError(f"Input lengths must all match; got {lengths}")
    if n < 3:
        raise ValueError(f"Need ≥ 3 complexes for ridge fit; got {n}")

    X = np.column_stack([
        np.array(vina_scores, dtype=float),
        np.array(ad4_scores, dtype=float),
        -np.array(n_contact_residues_list, dtype=float),
    ])
    y = np.array([-_RT * _LN10 * p for p in experimental_pkd], dtype=float)

    model = Ridge(alpha=ridge_alpha, positive=positive).fit(X, y)
    pred = model.predict(X)
    r_in = float(pearsonr(y, pred).statistic)
    rmse_in = float(np.sqrt(((pred - y) ** 2).mean()))

    # Leave-one-out CV
    loo_preds = np.zeros_like(y)
    for tr, te in LeaveOneOut().split(X):
        loo_preds[te] = (
            Ridge(alpha=ridge_alpha, positive=positive).fit(X[tr], y[tr]).predict(X[te])
        )
    r_loo = float(pearsonr(y, loo_preds).statistic)
    rmse_loo = float(np.sqrt(((loo_preds - y) ** 2).mean()))

    w_vina, w_ad4, w_neg_nc = (float(c) for c in model.coef_)
    intercept = float(model.intercept_)

    return {
        "schema_version": 2,
        "model_type": "ridge",
        "w_vina": w_vina,
        "w_ad4": w_ad4,
        "w_contact": -w_neg_nc,   # sign-flip back to "positive = entropy penalty"
        "intercept": intercept,
        "ridge_alpha": ridge_alpha,
        "positive_constraint": positive,
        "pearson_r": r_in,
        "rmse_kcal_mol": rmse_in,
        "loo_pearson_r": r_loo,
        "loo_rmse_kcal_mol": rmse_loo,
        "n_complexes": n,
    }


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
        alpha: Burial correction coefficient (kcal/mol/contact-residue).
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
