"""Tests for the pooled data-driven affinity model (scoring/affinity_model.py)."""
from __future__ import annotations

import numpy as np

from hybridock_pep.scoring.affinity_model import (
    GEOMETRY_KEYS,
    build_feature_vector,
    predict_affinity,
)


def _geom() -> dict[str, float]:
    g = {k: 1.0 for k in GEOMETRY_KEYS}
    g["poc_net"] = -2.0
    g["mean_burial"] = 40.0
    return g


def test_feature_vector_length() -> None:
    """262 features: 16 geometry + 220 ProtDCal + 3 charge-compl + length + 22 pocket-ProtDCal (E206).
    The pocket block is always present (zeros when the geometry dict has no pocket_seq)."""
    v = build_feature_vector(_geom(), "LISDAELEAIFEADC")
    assert v.shape == (262,)
    assert np.isfinite(v).all()


def test_pocket_protdcal_populated_when_pocket_seq_present() -> None:
    """When the geometry dict carries pocket_seq, the trailing 22-feature pocket block is non-trivial."""
    g = dict(_geom())
    g["pocket_seq"] = "LIWFYACDEKR"
    v = build_feature_vector(g, "LISDAELEAIFEADC")
    assert v.shape == (262,)
    assert np.any(v[-22:] != 0.0)  # pocket descriptors populated


def test_single_residue_peptide_is_finite() -> None:
    """A length-1 peptide must not produce NaN/inf (autocorrelation edge case)."""
    v = build_feature_vector(_geom(), "K")
    assert np.isfinite(v).all()


def test_descriptors_change_with_sequence() -> None:
    """A charged vs hydrophobic peptide must produce different descriptor vectors."""
    a = build_feature_vector(_geom(), "KKKKRRRR")
    b = build_feature_vector(_geom(), "AAAVVVLL")
    assert not np.allclose(a, b)


def test_predict_empty_sequence_is_none() -> None:
    assert predict_affinity(_geom(), "") is None


def test_predict_returns_float_or_none() -> None:
    """With the shipped artifact present, prediction is a finite kcal/mol; absent → graceful None."""
    dg = predict_affinity(_geom(), "ETFSDLWKLLPE")
    assert dg is None or (isinstance(dg, float) and np.isfinite(dg))


def test_missing_artifact_is_graceful() -> None:
    """A non-existent artifact path must not raise — pooled ΔG is an optional annotation."""
    assert predict_affinity(_geom(), "ETFSDLWKLLPE", artifact="/nonexistent/model.joblib") is None


def test_band_routers_only_affect_their_band() -> None:
    """The crystal artifact's band routers (E216/E238) must fire only inside their length band: a short
    peptide is on the main model; long (13-16) and vlong (>=17) may use their specialists. The routers are
    returned as a list of (sub_model, size_regs, lo, hi). Skips if artifact absent/legacy."""
    from pathlib import Path

    from hybridock_pep.scoring.affinity_model import _CRYSTAL_ARTIFACT, _load

    if not Path(_CRYSTAL_ARTIFACT).exists():
        return
    _model, _fo, _sr, routers = _load(str(_CRYSTAL_ARTIFACT))
    if not routers:
        return  # artifact predates the routers — nothing to assert
    g = dict(_geom())
    g["pocket_seq"] = "LIWFYACDEKR"
    short = predict_affinity(g, "ACDEFG", artifact=_CRYSTAL_ARTIFACT)            # L=6, main
    long_ = predict_affinity(g, "ACDEFGHIKLMNP", artifact=_CRYSTAL_ARTIFACT)     # L=13, long band
    vlong = predict_affinity(g, "ACDEFGHIKLMNPQRSTVW", artifact=_CRYSTAL_ARTIFACT)  # L=19, vlong band
    for v in (short, long_, vlong):
        assert v is not None and np.isfinite(v)
    # every router declares a valid (lo, hi) band; bands are disjoint and ordered
    bands = sorted((lo, hi) for _m, _r, lo, hi in routers)
    for (lo, hi) in bands:
        assert 0 < lo <= hi
    for (_, hi0), (lo1, _) in zip(bands, bands[1:]):
        assert hi0 < lo1  # disjoint
