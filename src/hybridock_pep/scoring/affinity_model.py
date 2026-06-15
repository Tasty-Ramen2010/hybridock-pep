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
# Sign-stable anchor features (commit cbd193bff, restored E170): help long/med +0.03–0.04 on pooled data.
# Appended only when present in the geometry dict (i.e. compute_anchor_features was run for the pose) so
# older 240-feature artifacts keep working unchanged.
ANCHOR_KEYS = ["max_burial", "buried_inert", "pro_run"]
# Size-confound fix (E201/E202/E203): the model over-relied on peptide length (corr(pred,len)=−0.21 vs
# truth −0.10), which sabotaged vlong (geometry features are size-driven and band-miscalibrated). These 7
# geometry features carry the size signal; the size-fix artifacts store length→feature regressors and
# residualise them at predict time, fixing vlong on BOTH crystal (0.07→0.16) and deployment (0.23→0.33).
SIZE_GEO_KEYS = ["poc_n", "bsa_hyd", "sasa_hb", "sasa_sb", "mj_contact", "mean_burial", "strength_bur"]
SIZE_IDX = [GEOMETRY_KEYS.index(k) for k in SIZE_GEO_KEYS]
_AA = "ACDEFGHIKLMNPQRSTVWY"
_POS, _NEG = set("KR"), set("DE")
_KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2,
       "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
       "Y": -1.3, "V": 4.2}
_PKA = {"D": 3.65, "E": 4.25, "H": 6.0, "C": 8.3, "Y": 10.1, "K": 10.5, "R": 12.5}

# ProtDCal-scale descriptor pool (E150): 22 amino-acid property scales × 10 aggregation operators = 220
# sequence descriptors. Built to close the charged gap vs PPI-Affinity (which selects 37 of ProtDCal's
# 23040); the hand-made 29-descriptor set was the limiting factor, not the data (E150/E151).
_SCALES = {
    "kd": {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2},
    "eisen": {"A": 0.62, "R": -2.53, "N": -0.78, "D": -0.9, "C": 0.29, "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.4, "I": 1.38, "L": 1.06, "K": -1.5, "M": 0.64, "F": 1.19, "P": 0.12, "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08},
    "hopp": {"A": -0.5, "R": 3.0, "N": 0.2, "D": 3.0, "C": -1.0, "Q": 0.2, "E": 3.0, "G": 0.0, "H": -0.5, "I": -1.8, "L": -1.8, "K": 3.0, "M": -1.3, "F": -2.5, "P": 0.0, "S": 0.3, "T": -0.4, "W": -3.4, "Y": -2.3, "V": -1.5},
    "charge": {a: (1.0 if a in "KR" else -1.0 if a in "DE" else 0.5 if a == "H" else 0.0) for a in _AA},
    "vol": {"A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0},
    "polar": {"A": 0.046, "R": 0.291, "N": 0.134, "D": 0.105, "C": 0.128, "Q": 0.18, "E": 0.151, "G": 0.0, "H": 0.23, "I": 0.186, "L": 0.186, "K": 0.219, "M": 0.221, "F": 0.29, "P": 0.131, "S": 0.062, "T": 0.108, "W": 0.409, "Y": 0.298, "V": 0.14},
    "pol_grantham": {"A": 8.1, "R": 10.5, "N": 11.6, "D": 13.0, "C": 5.5, "Q": 10.5, "E": 12.3, "G": 9.0, "H": 10.4, "I": 5.2, "L": 4.9, "K": 11.3, "M": 5.7, "F": 5.2, "P": 8.0, "S": 9.2, "T": 8.6, "W": 5.4, "Y": 6.2, "V": 5.9},
    "mw": {"A": 89, "R": 174, "N": 132, "D": 133, "C": 121, "Q": 146, "E": 147, "G": 75, "H": 155, "I": 131, "L": 131, "K": 146, "M": 149, "F": 165, "P": 115, "S": 105, "T": 119, "W": 204, "Y": 181, "V": 117},
    "bulk": {"A": 11.5, "R": 14.28, "N": 12.28, "D": 11.68, "C": 13.46, "Q": 14.45, "E": 13.57, "G": 3.4, "H": 13.69, "I": 21.4, "L": 21.4, "K": 15.71, "M": 16.25, "F": 19.8, "P": 17.43, "S": 9.47, "T": 15.77, "W": 21.67, "Y": 18.03, "V": 21.57},
    "flex": {"A": 0.36, "R": 0.53, "N": 0.46, "D": 0.51, "C": 0.35, "Q": 0.49, "E": 0.5, "G": 0.54, "H": 0.32, "I": 0.46, "L": 0.37, "K": 0.47, "M": 0.3, "F": 0.31, "P": 0.51, "S": 0.51, "T": 0.44, "W": 0.31, "Y": 0.42, "V": 0.39},
    "helix": {"A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.7, "Q": 1.11, "E": 1.51, "G": 0.57, "H": 1.0, "I": 1.08, "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57, "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06},
    "sheet": {"A": 0.83, "R": 0.93, "N": 0.89, "D": 0.54, "C": 1.19, "Q": 1.1, "E": 0.37, "G": 0.75, "H": 0.87, "I": 1.6, "L": 1.3, "K": 0.74, "M": 1.05, "F": 1.38, "P": 0.55, "S": 0.75, "T": 1.19, "W": 1.37, "Y": 1.47, "V": 1.7},
    "asa": {"A": 115, "R": 225, "N": 160, "D": 150, "C": 135, "Q": 180, "E": 190, "G": 75, "H": 195, "I": 175, "L": 170, "K": 200, "M": 185, "F": 210, "P": 145, "S": 115, "T": 140, "W": 255, "Y": 230, "V": 155},
    "refract": {"A": 4.34, "R": 26.66, "N": 13.28, "D": 12.0, "C": 35.77, "Q": 17.56, "E": 17.26, "G": 0.0, "H": 21.81, "I": 19.06, "L": 18.78, "K": 21.29, "M": 21.64, "F": 29.4, "P": 10.93, "S": 6.35, "T": 11.01, "W": 42.53, "Y": 31.53, "V": 13.92},
    "pI": {"A": 6.0, "R": 10.76, "N": 5.41, "D": 2.77, "C": 5.07, "Q": 5.65, "E": 3.22, "G": 5.97, "H": 7.59, "I": 6.02, "L": 5.98, "K": 9.74, "M": 5.74, "F": 5.48, "P": 6.3, "S": 5.68, "T": 5.6, "W": 5.89, "Y": 5.66, "V": 5.96},
    "transfer": {"A": 0.5, "R": -11.2, "N": -0.2, "D": -7.4, "C": -2.8, "Q": -9.38, "E": -9.9, "G": 0.0, "H": -0.5, "I": 2.5, "L": 1.8, "K": -4.2, "M": 1.3, "F": 2.5, "P": -3.3, "S": -0.3, "T": -0.4, "W": 3.4, "Y": 2.3, "V": 1.5},
    "isa": {"A": 0.31, "R": -1.01, "N": -0.6, "D": -0.77, "C": 1.54, "Q": -0.22, "E": -0.64, "G": 0.0, "H": 0.13, "I": 1.8, "L": 1.7, "K": -0.99, "M": 1.23, "F": 1.79, "P": 0.72, "S": -0.04, "T": 0.26, "W": 2.25, "Y": 0.96, "V": 1.22},
    "nci": {"A": 0.007, "R": 0.043, "N": -0.014, "D": -0.024, "C": 0.038, "Q": -0.011, "E": -0.012, "G": 0.018, "H": -0.04, "I": 0.022, "L": 0.052, "K": 0.018, "M": 0.003, "F": 0.038, "P": 0.24, "S": -0.005, "T": 0.003, "W": 0.05, "Y": 0.023, "V": 0.057},
    "alpha_n": {"A": 0.42, "R": 0.36, "N": 0.21, "D": 0.25, "C": 0.17, "Q": 0.36, "E": 0.42, "G": 0.13, "H": 0.27, "I": 0.3, "L": 0.39, "K": 0.32, "M": 0.38, "F": 0.3, "P": 0.13, "S": 0.2, "T": 0.21, "W": 0.32, "Y": 0.25, "V": 0.27},
    "hbond": {a: (1.0 if a in "STNQYHKRWDE" else 0.0) for a in _AA},
    "arom": {a: (1.0 if a in "FWY" else 0.0) for a in _AA},
    "sidechain_vol": {"A": 27, "R": 105, "N": 58, "D": 52, "C": 44, "Q": 80, "E": 73, "G": 0, "H": 79, "I": 93, "L": 93, "K": 100, "M": 94, "F": 115, "P": 41, "S": 29, "T": 51, "W": 145, "Y": 117, "V": 67},
}

# Two SEPARATE scoring functions, each tuned to its regime (E203/E204):
#   - AI / deployment (DEFAULT): trained on real RAPiDock poses, NO size-fix (data/affinity_ai_nofix.joblib).
#     The pipeline scores generated poses, so this is the default; the crystal model COLLAPSES on real poses
#     (E152 "AI haircut"). The size-fix HELPS crystal but HURTS deployment on current data (real poses carry
#     pose-quality signal in the geometry block), so the AI model deliberately omits it.
#   - CRYSTAL: trained on crystal-925 WITH the size-fix (data/affinity_crystal_sizefix.joblib) — use only for
#     crystal inputs. The size-fix residualises size-geometry vs length, fixing crystal vlong 0.07→0.16 and
#     lifting short 0.46→0.49 (E203), and is applied at predict time via the artifact's size_regs.
# Artifacts without size_regs (the AI model, plus legacy artifacts) load with no residualisation.
_DEFAULT_ARTIFACT = Path(__file__).resolve().parents[3] / "data" / "affinity_ai_nofix.joblib"
_CRYSTAL_ARTIFACT = Path(__file__).resolve().parents[3] / "data" / "affinity_crystal_sizefix.joblib"


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


def _protdcal_descriptors(seq: str) -> list[float]:
    """220 ProtDCal-scale descriptors: 22 property scales × 10 aggregations (mean/std/max/min/sum/range/
    Nterm/Cterm/autocorr-lag1/lag2). NaN-safe (constant scales → 0 autocorrelation)."""
    out: list[float] = []
    for sc in _SCALES.values():
        v = np.array([sc.get(c, 0.0) for c in seq], dtype=float)
        if v.size == 0:
            out += [0.0] * 10
            continue
        ac1 = float(np.corrcoef(v[:-1], v[1:])[0, 1]) if v.size > 2 and np.std(v) > 0 else 0.0
        ac2 = float(np.corrcoef(v[:-2], v[2:])[0, 1]) if v.size > 3 and np.std(v) > 0 else 0.0
        out += [v.mean(), v.std(), v.max(), v.min(), v.sum(), v.max() - v.min(),
                v[0], v[-1], 0.0 if np.isnan(ac1) else ac1, 0.0 if np.isnan(ac2) else ac2]
    return out


def _charge_complementarity(seq: str, poc_net: float) -> list[float]:
    """Peptide×pocket net charge complementarity (the electrostatics that does not wash, E149)."""
    pq = sum(c in _POS for c in seq) - sum(c in _NEG for c in seq)
    return [float(pq * poc_net), float(abs(pq) * abs(poc_net)), float(abs(pq + poc_net))]


@lru_cache(maxsize=4)
def _load(artifact: str):
    try:
        import joblib
        bundle = joblib.load(artifact)
        # size_regs: {geometry_index: [coef, intercept]} for the length→size-feature residualisation.
        return bundle["model"], bundle.get("feature_order"), bundle.get("size_regs")
    except FileNotFoundError:
        logger.warning("Affinity model: artifact not found at %s — pooled ΔG skipped", artifact)
        return None, None, None
    except Exception as exc:  # noqa: BLE001 — never break the pipeline on an optional annotation
        logger.warning("Affinity model: failed to load %s (%s) — pooled ΔG skipped", artifact, exc)
        return None, None, None


def _apply_size_fix(x: np.ndarray, seq_len: int, size_regs: dict | None) -> np.ndarray:
    """Residualise the size-correlated geometry features against peptide length (E203 size-fix).

    Removes the length-predictable component of each size-geometry feature so the model cannot over-rely on
    size. No-op for legacy artifacts that carry no ``size_regs``. The geometry block sits at indices 0..15,
    unaffected by the optional trailing anchor block, so this is safe before any length trim.
    """
    if not size_regs:
        return x
    x = x.copy()
    for j, (coef, intercept) in size_regs.items():
        j = int(j)
        if j < x.shape[0]:
            x[j] = x[j] - (coef * float(seq_len) + intercept)
    return x


def build_feature_vector(geometry: dict[str, float], seq: str) -> np.ndarray:
    """Assemble the 49-feature production vector from geometry descriptors + peptide sequence.

    Args:
        geometry: dict with the 16 GEOMETRY_KEYS (from ``compute_geometry_features``); ``poc_net`` is also
            reused for charge complementarity.
        seq: one-letter peptide sequence (length drives the soft per-band conditioning).

    Returns:
        Length-240 float array in the model's training order
        (16 geometry + 220 ProtDCal + 3 charge-complementarity + length).
    """
    geom = [float(geometry.get(k, 0.0)) for k in GEOMETRY_KEYS]
    pdesc = _protdcal_descriptors(seq)
    compl = _charge_complementarity(seq, float(geometry.get("poc_net", 0.0)))
    # Restored anchor features are appended only when the pose's geometry dict carries them, so artifacts
    # trained without anchor (240-feat) and with anchor (243-feat) both consume the matching vector length.
    anchor = [float(geometry[k]) for k in ANCHOR_KEYS] if all(k in geometry for k in ANCHOR_KEYS) else []
    vec = np.asarray(geom + pdesc + compl + [float(len(seq))] + anchor, dtype=float)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


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
    model, _, size_regs = _load(str(artifact or _DEFAULT_ARTIFACT))
    if model is None:
        return None
    x = build_feature_vector(geometry, seq)
    # Size-confound fix (E203): residualise size-geometry vs length before predicting (no-op for legacy
    # artifacts without size_regs). Applied to the stable geometry block, before any anchor trim.
    x = _apply_size_fix(x, len(seq), size_regs)
    # Backward-compatible feature length: build_feature_vector appends the 3 anchor features when the pose
    # carries them, producing 243. Models trained without anchor expect 240 — trim the trailing anchor block
    # so old (240) and anchor (243) artifacts both work regardless of what the driver computed.
    n_exp = getattr(model, "n_features_in_", x.shape[0])
    if x.shape[0] > n_exp:
        x = x[:n_exp]
    return float(model.predict(x.reshape(1, -1))[0])
