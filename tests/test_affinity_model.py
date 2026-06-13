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
    """240 features: 16 geometry + 220 ProtDCal + 3 charge-compl + length."""
    v = build_feature_vector(_geom(), "LISDAELEAIFEADC")
    assert v.shape == (240,)
    assert np.isfinite(v).all()


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
