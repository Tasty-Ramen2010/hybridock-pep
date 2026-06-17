"""Same-receptor reference anchoring — calibrate an absolute ΔG to known-Kd peptides on the SAME target.

Deployment of the validated reference-anchoring result (see docs/reference_anchoring_design.md). The
absolute scorer carries a per-receptor offset ``b(R)`` that is FEP-bound and unpredictable from any static
feature (sequence, pocket sequence, pocket-3D — all tested, e266–e273). It CANCELS, however, when the
reference peptide sits on the SAME receptor as the query::

    ΔG_pred(P, R) = Σ_k w_k [ ΔG_exp(ref_k) + S(P, R) − S(ref_k, R) ]

where ``S`` is the absolute scorer and ``w_k`` weights references by peptide-feature similarity. On real
peptide Kd this lifts within-receptor accuracy from r≈0.26 (absolute) to r≈0.63, MAE 2.09→1.65 kcal/mol
(e274). It is a SAME-RECEPTOR few-shot calibrator: cross-receptor transfer does NOT work (the offset does
not transfer by any similarity metric, and averaging cannot remove a systematic bias). When no
same-receptor reference exists, this module returns the absolute score unchanged with ``mode="absolute"``.

Honest scope:
  * Needs ≥1 known-Kd peptide on the query receptor (or a ≥``IDENTITY_FLOOR`` near-identical sequence).
  * Accuracy is capped at the within-receptor η floor (~1.3–1.6 kcal/mol RMSE); it is a calibrator, not
    an FEP replacement.
  * Do NOT anchor across different receptors — pass only same-receptor references.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Sequence identity at/above which two receptor sequences are treated as the SAME target for anchoring.
IDENTITY_FLOOR = 0.90


@dataclass(frozen=True)
class Reference:
    """A known-Kd reference peptide on a specific receptor.

    Attributes:
        peptide: Reference peptide one-letter sequence.
        receptor: Receptor one-letter sequence the reference Kd was measured on.
        dg_exp: Experimental binding free energy in kcal/mol (negative = binding).
        features: Peptide feature vector used for similarity weighting (same space as the query).
        score: Absolute scorer output ``S(ref, R)`` for this reference, in kcal/mol.
    """

    peptide: str
    receptor: str
    dg_exp: float
    features: np.ndarray
    score: float


@dataclass(frozen=True)
class AnchorResult:
    """Outcome of an anchoring attempt.

    Attributes:
        dg: Predicted binding free energy in kcal/mol.
        mode: ``"anchored"`` if a same-receptor reference was used, else ``"absolute"``.
        n_refs: Number of same-receptor references used (0 in absolute mode).
        confidence: Heuristic 0–1 confidence; rises with reference count and peptide similarity.
    """

    dg: float
    mode: str
    n_refs: int
    confidence: float


def _kmers(seq: str, k: int = 4) -> set[str]:
    return {seq[i : i + k] for i in range(len(seq) - k + 1)} if len(seq) >= k else set()


def sequence_identity(a: str, b: str) -> float:
    """K-mer Jaccard similarity of two sequences (a cheap proxy for sequence identity).

    Args:
        a: First sequence.
        b: Second sequence.

    Returns:
        Jaccard overlap in ``[0, 1]``; ``1.0`` for identical sequences, ``0.0`` if either is too short.
    """
    ka, kb = _kmers(a), _kmers(b)
    if not ka or not kb:
        return 1.0 if a == b else 0.0
    return len(ka & kb) / len(ka | kb)


def anchored_affinity(
    receptor: str,
    query_features: np.ndarray,
    absolute_dg: float,
    references: list[Reference],
    *,
    sigma: float | None = None,
    identity_floor: float = IDENTITY_FLOOR,
) -> AnchorResult:
    """Calibrate an absolute ΔG using known-Kd peptides on the same receptor.

    Args:
        receptor: Query receptor one-letter sequence.
        query_features: Query peptide feature vector (same space as ``Reference.features``).
        absolute_dg: The absolute scorer output ``S(P, R)`` for the query, in kcal/mol.
        references: Candidate references; only those on the same receptor (sequence identity
            ≥ ``identity_floor``) are used. Pass the full library; filtering happens here.
        sigma: Length scale for the similarity kernel in feature space. Defaults to the median
            pairwise distance among the matched references (robust, scale-free).
        identity_floor: Minimum receptor sequence identity to treat a reference as same-target.

    Returns:
        An :class:`AnchorResult`. Falls back to ``mode="absolute"`` (``dg=absolute_dg``) when no
        same-receptor reference is available.

    Raises:
        ValueError: If ``query_features`` is empty or any reference feature dimension mismatches.
    """
    query_features = np.asarray(query_features, dtype=float)
    if query_features.size == 0:
        raise ValueError("query_features must be non-empty")

    matched = [r for r in references if sequence_identity(receptor, r.receptor) >= identity_floor]
    if not matched:
        logger.debug("anchoring: no same-receptor reference; falling back to absolute ΔG")
        return AnchorResult(dg=float(absolute_dg), mode="absolute", n_refs=0, confidence=0.0)

    feats = np.array([r.features for r in matched], dtype=float)
    if feats.shape[1] != query_features.shape[0]:
        raise ValueError(
            f"reference feature dim {feats.shape[1]} != query dim {query_features.shape[0]}"
        )

    dist = np.linalg.norm(feats - query_features, axis=1)
    if sigma is None:
        sigma = float(np.median(dist)) or 1.0
    # log-domain softmax for numerical stability (far references underflow gracefully)
    logw = -(dist**2) / (2.0 * sigma**2)
    logw -= logw.max()
    w = np.exp(logw)
    w /= w.sum()

    dg_exp = np.array([r.dg_exp for r in matched])
    score = np.array([r.score for r in matched])
    dg = float(np.sum(w * (dg_exp + absolute_dg - score)))

    # confidence: saturating in #refs, scaled by closeness of the nearest reference
    nearest = float(np.exp(-dist.min() / sigma))
    confidence = float((1.0 - np.exp(-len(matched) / 3.0)) * nearest)
    logger.debug(
        "anchoring: %d same-receptor refs, ΔG %.2f→%.2f kcal/mol (conf %.2f)",
        len(matched),
        absolute_dg,
        dg,
        confidence,
    )
    return AnchorResult(dg=dg, mode="anchored", n_refs=len(matched), confidence=confidence)
