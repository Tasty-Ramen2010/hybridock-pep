"""Unit tests for same-receptor reference anchoring (scoring/anchoring.py)."""
from __future__ import annotations

import numpy as np
import pytest

from hybridock_pep.scoring.anchoring import (
    AnchorResult,
    Reference,
    anchored_affinity,
    sequence_identity,
)

RECEPTOR = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
OTHER = "GSHMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def _ref(receptor: str, dg: float, feats: list[float], score: float) -> Reference:
    return Reference(peptide="PEP", receptor=receptor, dg_exp=dg,
                     features=np.array(feats, float), score=score)


def test_no_same_receptor_falls_back_to_absolute() -> None:
    """With only different-receptor references, anchoring returns the absolute ΔG unchanged."""
    refs = [_ref(OTHER, dg=-9.0, feats=[1.0, 2.0], score=-7.0)]
    res = anchored_affinity(RECEPTOR, np.array([1.0, 2.0]), absolute_dg=-6.0, references=refs)
    assert res.mode == "absolute"
    assert res.n_refs == 0
    assert res.dg == pytest.approx(-6.0)
    assert res.confidence == 0.0


def test_same_receptor_anchors_and_cancels_offset() -> None:
    """A same-receptor reference shifts the prediction by the reference's (ΔG_exp − score) offset.

    With one reference, ΔG_pred = ΔG_exp(ref) + S(query) − S(ref). Here the scorer is biased +3 on this
    receptor (score −7 vs true −10); anchoring removes that bias.
    """
    refs = [_ref(RECEPTOR, dg=-10.0, feats=[1.0, 2.0], score=-7.0)]
    res = anchored_affinity(RECEPTOR, np.array([1.05, 2.05]), absolute_dg=-7.2, references=refs)
    assert res.mode == "anchored"
    assert res.n_refs == 1
    # -10.0 + (-7.2) - (-7.0) = -10.2  (the +3 offset is cancelled)
    assert res.dg == pytest.approx(-10.2, abs=1e-6)
    assert res.confidence > 0.0


def test_similar_reference_weighted_more() -> None:
    """The nearer-in-feature reference dominates the weighted anchor."""
    refs = [
        _ref(RECEPTOR, dg=-12.0, feats=[1.0, 1.0], score=-12.0),   # far from query
        _ref(RECEPTOR, dg=-8.0, feats=[5.0, 5.0], score=-8.0),     # near query
    ]
    res = anchored_affinity(RECEPTOR, np.array([4.9, 4.9]), absolute_dg=-8.0, references=refs)
    # offsets are zero for both refs, so dg≈absolute, but the near ref (-8 home) must dominate weighting
    assert res.mode == "anchored"
    assert res.n_refs == 2
    assert res.dg == pytest.approx(-8.0, abs=0.5)


def test_identity_floor_excludes_distant_homologs() -> None:
    """A merely-similar (below floor) receptor is not treated as the same target."""
    near_identical = RECEPTOR[:-2] + "AA"  # ~97% identical -> same target
    distant = RECEPTOR[:20] + OTHER[20:]   # chimeric -> below floor
    refs = [_ref(distant, dg=-9.0, feats=[1.0, 2.0], score=-9.0)]
    res = anchored_affinity(RECEPTOR, np.array([1.0, 2.0]), absolute_dg=-6.0, references=refs)
    assert res.mode == "absolute"
    assert sequence_identity(RECEPTOR, near_identical) >= 0.90
    assert sequence_identity(RECEPTOR, distant) < 0.90


def test_feature_dim_mismatch_raises() -> None:
    refs = [_ref(RECEPTOR, dg=-9.0, feats=[1.0, 2.0, 3.0], score=-9.0)]
    with pytest.raises(ValueError):
        anchored_affinity(RECEPTOR, np.array([1.0, 2.0]), absolute_dg=-6.0, references=refs)


def test_empty_query_features_raises() -> None:
    with pytest.raises(ValueError):
        anchored_affinity(RECEPTOR, np.array([]), absolute_dg=-6.0, references=[])
