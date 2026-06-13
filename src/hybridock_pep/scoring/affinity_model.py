"""Pooled data-driven affinity model — the length-conditioned, descriptor-augmented production scorer.

Trained on 1076 pooled peptide–protein complexes (PDBbind-925 + curated benchmark) over 49 features:
the 16 geometry descriptors (``geometry_features``) + 29 sequence physicochemical descriptors + 3
peptide×pocket charge-complementarity terms + peptide length. Grouped-CV r≈0.51 overall (MAE 1.31),
short≈0.50, charged≈0.43; on the curated benchmark r≈0.58 / MAE 1.41 — matches PPI-Affinity on correlation
and beats it on MAE (their reported metric, ~1.8).

Design notes:
- Length is a FEATURE (soft per-band conditioning), not a hard router — hard routing starves bands (E126).
- Sequence descriptors recover part of the charged floor that single-pose physics electrostatics wash out
  (E146/E149): the charged signal is partly data-learnable, as PPI-Affinity demonstrates.
- Graceful no-op: if the artifact is absent the scorer returns None and the pipeline annotation is skipped.

Artifact: ``data/affinity_pooled_prodn.joblib`` (dict: model, feature_order, n_train).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# 16 geometry features, in the order the model was trained (matches geometry_features + mean_burial).
GEOMETRY_KEYS = [
    "poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
    "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density", "cys_frac",
]
_AA = "ACDEFGHIKLMNPQRSTVWY"
_POS, _NEG = set("KR"), set("DE")
_KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2,
       "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
       "Y": -1.3, "V": 4.2}
_PKA = {"D": 3.65, "E": 4.25, "H": 6.0, "C": 8.3, "Y": 10.1, "K": 10.5, "R": 12.5}

_DEFAULT_ARTIFACT = Path(__file__).resolve().parents[3] / "data" / "affinity_pooled_prodn.joblib"


def _approx_pI(seq: str) -> float:
    """Approximate isoelectric point by bisection on the Henderson–Hasselbalch net charge."""
    def charge(ph: float) -> float:
        c = 1 / (1 + 10 ** (ph - 8.0)) - 1 / (1 + 10 ** (3.1 - ph))  # N/C termini
        for a in seq:
            if a in ("K", "R", "H"):
                c += 1 / (1 + 10 ** (ph - _PKA[a]))
            elif a in ("D", "E", "C", "Y"):
                c -= 1 / (1 + 10 ** (_PKA[a] - ph))
        return c
    lo, hi = 0.0, 14.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if charge(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _rich_descriptors(seq: str) -> list[float]:
    """29 sequence physicochemical descriptors (charge pattern, pI, amphipathy, composition)."""
    L = max(1, len(seq))
    npos = sum(c in _POS for c in seq)
    nneg = sum(c in _NEG for c in seq)
    pos_idx = [i for i, c in enumerate(seq) if c in _POS]
    neg_idx = [i for i, c in enumerate(seq) if c in _NEG]

    def clust(idx: list[int]) -> float:
        return float(np.mean(np.diff(sorted(idx))) / L) if len(idx) > 1 else 1.0

    ang = np.arange(L) * (100 * np.pi / 180)          # Eisenberg helical hydrophobic moment
    h = np.array([_KD.get(c, 0.0) for c in seq])
    hm = float(np.sqrt((h * np.cos(ang)).sum() ** 2 + (h * np.sin(ang)).sum() ** 2) / L)
    comp = [seq.count(a) / L for a in _AA]
    return [float(npos - nneg), float(abs(npos - nneg)), _approx_pI(seq), clust(pos_idx), clust(neg_idx),
            hm, (npos + nneg) / L, float(npos), float(nneg)] + comp


def _charge_complementarity(seq: str, poc_net: float) -> list[float]:
    """Peptide×pocket net charge complementarity (the electrostatics that does not wash, E149)."""
    pq = sum(c in _POS for c in seq) - sum(c in _NEG for c in seq)
    return [float(pq * poc_net), float(abs(pq) * abs(poc_net)), float(abs(pq + poc_net))]


@lru_cache(maxsize=4)
def _load(artifact: str):
    try:
        import joblib
        bundle = joblib.load(artifact)
        return bundle["model"], bundle.get("feature_order")
    except FileNotFoundError:
        logger.warning("Affinity model: artifact not found at %s — pooled ΔG skipped", artifact)
        return None, None
    except Exception as exc:  # noqa: BLE001 — never break the pipeline on an optional annotation
        logger.warning("Affinity model: failed to load %s (%s) — pooled ΔG skipped", artifact, exc)
        return None, None


def build_feature_vector(geometry: dict[str, float], seq: str) -> np.ndarray:
    """Assemble the 49-feature production vector from geometry descriptors + peptide sequence.

    Args:
        geometry: dict with the 16 GEOMETRY_KEYS (from ``compute_geometry_features``); ``poc_net`` is also
            reused for charge complementarity.
        seq: one-letter peptide sequence (length drives the soft per-band conditioning).

    Returns:
        Length-49 float array in the model's training order.
    """
    geom = [float(geometry.get(k, 0.0)) for k in GEOMETRY_KEYS]
    rich = _rich_descriptors(seq)
    compl = _charge_complementarity(seq, float(geometry.get("poc_net", 0.0)))
    return np.asarray(geom + rich + compl + [float(len(seq))], dtype=float)


def predict_affinity(geometry: dict[str, float], seq: str, artifact: Path | str | None = None) -> float | None:
    """Predict calibrated ΔG (kcal/mol) for one pose, or None if the model artifact is unavailable.

    Args:
        geometry: the 16 geometry descriptors for the pose.
        seq: peptide one-letter sequence.
        artifact: optional path to the joblib bundle; defaults to ``data/affinity_pooled_prodn.joblib``.

    Returns:
        Predicted ΔG in kcal/mol, or None if the artifact is missing/unloadable or seq is empty.
    """
    if not seq:
        return None
    model, _ = _load(str(artifact or _DEFAULT_ARTIFACT))
    if model is None:
        return None
    x = build_feature_vector(geometry, seq).reshape(1, -1)
    return float(model.predict(x)[0])
