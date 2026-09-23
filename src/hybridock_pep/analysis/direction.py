"""Peptide threading direction, and using a co-folded prior to filter on it.

THE PROBLEM THIS ADDRESSES
--------------------------
A pseudo-symmetric repeat groove -- armadillo, ankyrin, TPR, and the de novo designed
binders of Wu et al. 2025 that are built from exactly that architecture -- presents the same
chemistry looking N-to-C as it does C-to-N. Our diffusion sampler has no way to break that
degeneracy and threads the peptide backwards in 79-81% of poses. The pose is otherwise close:
centroid 0.9 A, Kabsch shape error 2.6 A, register-corrected RMSD 2.3-2.8 A. It is the
direction that is wrong.

Two things are established about it and both matter here:

  * It is not a sampling limit. Going N=24 -> 192 -> 500 moves best RMSD 6.74 -> 5.57 -> 5.66.
    More samples do not find the forward pose; they find more backward ones.
  * It is not recoverable by re-ranking. ref2015 prefers the forward pose 41.0% of the time,
    i.e. slightly worse than a coin, and the best of five geometric descriptors reaches
    AUC 0.603. Nothing in the energy function sees direction.

That leaves one option: get the direction from somewhere that does know it, and use it to
filter. This module is the filter. It does not care where the prior comes from -- a co-folded
complex, a homologous crystal structure, a user assertion -- only that it supplies an axis.

WHAT "DIRECTION" MEANS HERE
---------------------------
The N-to-C axis of the peptide CA trace, as a unit vector. For an extended peptide in a
groove this is well defined and nearly the whole story. For a peptide that folds back on
itself, or a short one that is more blob than strand, it is not, so `axis_quality` reports
the straightness and the caller should not filter on a prior whose own axis is ill-defined.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

#: A pose whose direction cosine with the prior is below this is threading the groove the
#: other way. 0.0 is the natural cut -- it is the sign of the projection that matters, not
#: its magnitude -- but a small positive margin avoids keeping poses that sit right on the
#: perpendicular, where the axis carries no information either way.
DEFAULT_COS_CUT = 0.0
#: Below this straightness the peptide is not extended enough for an N-to-C axis to mean
#: anything, and direction filtering is refused rather than applied to noise.
MIN_AXIS_QUALITY = 0.55


def axis(ca: np.ndarray) -> np.ndarray:
    """Unit vector from the N-terminal CA to the C-terminal CA.

    Args:
        ca: (n_res, 3) CA coordinates in N-to-C order.

    Returns:
        Unit 3-vector, or a zero vector if the termini coincide.

    Raises:
        ValueError: If fewer than two residues are given.
    """
    if len(ca) < 2:
        raise ValueError(f"need >=2 residues for an axis, got {len(ca)}")
    v = np.asarray(ca[-1], dtype=float) - np.asarray(ca[0], dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.zeros(3)


def axis_quality(ca: np.ndarray) -> float:
    """How well a single N-to-C axis describes this peptide's path, in [0, 1].

    End-to-end distance divided by contour length along the CA trace. An ideal extended
    strand is ~1.0; a helix is ~0.45; a hairpin that folds back is near 0.

    Args:
        ca: (n_res, 3) CA coordinates in N-to-C order.

    Returns:
        Straightness ratio. 0.0 if the trace has no length.
    """
    ca = np.asarray(ca, dtype=float)
    contour = float(np.linalg.norm(np.diff(ca, axis=0), axis=1).sum())
    if contour < 1e-6:
        return 0.0
    return float(np.linalg.norm(ca[-1] - ca[0]) / contour)


def agreement(pose_ca: np.ndarray, prior_axis: np.ndarray) -> float:
    """Cosine between a pose's N-to-C axis and the prior's.

    Args:
        pose_ca: (n_res, 3) CA coordinates of the candidate pose, N-to-C order.
        prior_axis: Unit 3-vector from the prior.

    Returns:
        Cosine in [-1, 1]. Near +1 the pose threads the groove the same way as the prior;
        near -1 it is threaded backwards.
    """
    a = axis(pose_ca)
    if not np.any(a) or not np.any(prior_axis):
        return 0.0
    return float(np.dot(a, prior_axis))


def filter_by_direction(
    poses: list[tuple[str, np.ndarray]],
    prior_axis: np.ndarray,
    prior_ca: np.ndarray | None = None,
    cos_cut: float = DEFAULT_COS_CUT,
    min_keep: int = 3,
) -> tuple[list[str], dict]:
    """Keep only poses that thread the peptide the same way as the prior.

    This is deliberately a filter and not a score. The prior is trusted for one bit -- which
    way round -- and nothing else, so it removes candidates rather than reordering them, and
    the surviving poses are still ranked by the same physics as always.

    Args:
        poses: (identifier, CA coordinates) for each candidate pose, N-to-C order.
        prior_axis: Unit direction vector from the prior.
        prior_ca: The prior's own CA trace, used to check its axis is meaningful. If None the
            check is skipped and the caller takes responsibility for it.
        cos_cut: Keep poses whose direction cosine with the prior exceeds this.
        min_keep: Never return fewer than this many poses. If the filter would, it is
            abandoned and every pose is kept -- an aggressive filter that leaves one
            candidate has replaced an ensemble with an assertion, which is worse than no
            filter at all.

    Returns:
        (kept identifiers, diagnostics dict with n_in, n_kept, applied, reason, cosines).
    """
    diag: dict = {"n_in": len(poses), "cos_cut": cos_cut, "applied": False}
    if prior_ca is not None:
        q = axis_quality(prior_ca)
        diag["prior_axis_quality"] = round(q, 3)
        if q < MIN_AXIS_QUALITY:
            diag["reason"] = (
                f"prior peptide is not extended enough (straightness {q:.2f} < "
                f"{MIN_AXIS_QUALITY}); its N-to-C axis does not define a threading direction"
            )
            logger.info("direction filter skipped: %s", diag["reason"])
            return [p[0] for p in poses], diag
    if not np.any(prior_axis):
        diag["reason"] = "prior axis is degenerate"
        return [p[0] for p in poses], diag

    cos = {pid: agreement(ca, prior_axis) for pid, ca in poses}
    diag["cosines"] = {k: round(v, 3) for k, v in cos.items()}
    kept = [pid for pid, c in cos.items() if c > cos_cut]
    diag["n_forward"] = len(kept)
    if len(kept) < min_keep:
        diag["reason"] = (
            f"only {len(kept)} of {len(poses)} poses agree with the prior direction, below "
            f"min_keep={min_keep}; keeping all rather than trusting the prior that far"
        )
        logger.warning("direction filter abandoned: %s", diag["reason"])
        return [p[0] for p in poses], diag

    diag["applied"] = True
    diag["n_kept"] = len(kept)
    logger.info(
        "direction filter: kept %d/%d poses threading the groove forward", len(kept), len(poses)
    )
    return kept, diag
