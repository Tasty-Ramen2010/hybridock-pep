"""Tests for the SKEMPI-trained single-mutation ΔΔG head."""
from __future__ import annotations

import numpy as np
import pytest

from hybridock_pep.scoring import mutation_ddg as m


def test_mutation_features_length_matches_artifact() -> None:
    """The feature vector must match the trained artifact's expected width."""
    art = m._load_model()
    feats = m.mutation_features("K", "A", "COR")
    if art is not None:
        assert len(feats) == len(art["feature_names"])
    assert len(feats) == 34 + len(m.LOCATIONS)


def test_mutation_features_are_antisymmetric_in_the_deltas() -> None:
    """Reversing a mutation must negate the delta block (the first 9 features)."""
    fwd = np.array(m.mutation_features("K", "D", "COR"))
    rev = np.array(m.mutation_features("D", "K", "COR"))
    np.testing.assert_allclose(fwd[:9], -rev[:9], atol=1e-9)


def test_identity_mutation_has_zero_deltas() -> None:
    """Mutating a residue to itself leaves every delta at zero."""
    feats = np.array(m.mutation_features("W", "W", "COR"))
    np.testing.assert_allclose(feats[:9], 0.0, atol=1e-9)


def test_location_one_hot_is_exclusive_and_unknown_is_all_zero() -> None:
    """Exactly one location bit is set for a known class; none for an unknown one."""
    n_loc = len(m.LOCATIONS)
    for loc in m.LOCATIONS:
        tail = m.mutation_features("K", "A", loc)[-n_loc:]
        assert sum(tail) == 1.0
    assert sum(m.mutation_features("K", "A", "NOT_A_CLASS")[-n_loc:]) == 0.0


@pytest.mark.parametrize("wt,mut", [("K", "Z"), ("X", "A"), ("K", "B")])
def test_non_standard_residue_raises(wt: str, mut: str) -> None:
    """Non-standard amino acids raise rather than silently scoring."""
    with pytest.raises(ValueError, match="non-standard amino acid"):
        m.mutation_features(wt, mut)


def test_predict_ddg_returns_finite_or_none() -> None:
    """A prediction is either a finite float or None when the artifact is missing."""
    val = m.predict_ddg("K", "A", "COR")
    assert val is None or np.isfinite(val)


def test_predictions_are_not_all_identical() -> None:
    """The head must actually discriminate between mutations, not emit a constant."""
    if m._load_model() is None:
        pytest.skip("SKEMPI artifact not installed")
    vals = [m.predict_ddg(wt, "A", "COR") for wt in "WFYLIVKRDEST"]
    assert np.std(vals) > 0.05, "ΔΔG head is emitting a near-constant"


def test_scan_variants_rejects_multi_and_length_mismatch() -> None:
    """Only true single substitutions of equal length are accepted."""
    if m._load_model() is None:
        pytest.skip("SKEMPI artifact not installed")
    with pytest.raises(ValueError, match="differs from reference at 2 positions"):
        m.scan_variants("KAK", ["KDD"])
    with pytest.raises(ValueError, match="length"):
        m.scan_variants("KAK", ["KAKA"])
    with pytest.raises(ValueError, match="differs from reference at 0 positions"):
        m.scan_variants("KAK", ["KAK"])


def test_scan_variants_sorted_and_positions_correct() -> None:
    """Results are sorted most-stabilising first and report the right position."""
    if m._load_model() is None:
        pytest.skip("SKEMPI artifact not installed")
    out = m.scan_variants("KAK", ["KDK", "KWK", "KVK"])
    assert [e.position for e in out] == [2, 2, 2]
    assert [e.wild_type for e in out] == ["A", "A", "A"]
    assert out == sorted(out, key=lambda e: e.ddg)
    assert all(e.expected_error > 0 for e in out)


def test_alanine_scan_skips_alanine_and_covers_the_rest() -> None:
    """Alanine positions are skipped; every other standard residue is scanned in order."""
    if m._load_model() is None:
        pytest.skip("SKEMPI artifact not installed")
    out = m.alanine_scan("KAWDA")
    assert [e.position for e in out] == [1, 3, 4]
    assert [e.wild_type for e in out] == ["K", "W", "D"]
    assert all(e.mutant == "A" for e in out)


def test_alanine_scan_all_alanine_returns_empty() -> None:
    """An all-alanine peptide has nothing to scan and returns an empty list, not None."""
    if m._load_model() is None:
        pytest.skip("SKEMPI artifact not installed")
    assert m.alanine_scan("AAAA") == []


def test_hot_spot_threshold_flag() -> None:
    """is_hot_spot reflects the documented threshold."""
    hot = m.MutationEffect(1, "W", "A", m.HOT_SPOT_THRESHOLD + 0.5, 1.0, "COR")
    cold = m.MutationEffect(1, "S", "A", m.HOT_SPOT_THRESHOLD - 0.5, 1.0, "COR")
    assert hot.is_hot_spot and not cold.is_hot_spot


def test_missing_artifact_degrades_to_none(tmp_path, monkeypatch) -> None:
    """A missing artifact disables the head rather than raising into a docking run."""
    monkeypatch.setattr(m, "_load_model", lambda artifact=None: None)
    assert m.predict_ddg("K", "A") is None
    assert m.scan_variants("KAK", ["KDK"]) is None
    assert m.alanine_scan("KDK") is None


def test_bulky_to_alanine_costs_more_than_small_to_alanine() -> None:
    """Sanity: deleting a large buried side chain should cost more than deleting a small one.

    This is a directional check on learned behaviour, not an exact value, so it is deliberately
    loose — it catches a sign flip or a scrambled feature order, which is what could silently break.
    """
    if m._load_model() is None:
        pytest.skip("SKEMPI artifact not installed")
    bulky = np.mean([m.predict_ddg(a, "A", "COR") for a in "WFY"])
    small = np.mean([m.predict_ddg(a, "A", "COR") for a in "SGT"])
    assert bulky > small, f"W/F/Y→A ({bulky:.2f}) should cost more than S/G/T→A ({small:.2f})"
