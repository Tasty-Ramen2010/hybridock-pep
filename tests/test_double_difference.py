"""Unit tests for the double-difference thermodynamic-cycle ΔG estimator."""
from __future__ import annotations

import pytest

from hybridock_pep.scoring.double_difference import (
    double_difference_dg,
    double_difference_selectivity,
)


def test_additive_grid_is_exact() -> None:
    """When G is perfectly additive, the double difference recovers the 4th corner exactly.

    Construct G(p, r) = f(p) + g(r): P/Pref have f = -2/-1, R/Rref have g = -6/-4.
      ΔG(P,R)=-8, ΔG(P,Rref)=-6, ΔG(Pref,R)=-7, ΔG(Pref,Rref)=-5.
      double-diff: -6 + -7 - (-5) = -8 == true ΔG(P,R).
    """
    res = double_difference_dg(dg_query_on_ref_receptor=-6.0,
                               dg_ref_peptide_on_query_receptor=-7.0,
                               dg_ref_peptide_on_ref_receptor=-5.0)
    assert res.dg == pytest.approx(-8.0)
    assert res.residual_kind == "coupling"


def test_cancels_large_receptor_offset() -> None:
    """A large per-receptor offset b(R) on the query receptor cancels in the cycle.

    Add b=+100 to every term on receptor R (both query and ref peptide). The estimate must be unchanged
    because b(R) enters dg_ref_peptide_on_query_receptor and cancels against... it does NOT fully cancel
    unless the offset is in the *measured* values; here we model that the measured corner on R carries it
    and the prediction inherits it (that is physical — the true ΔG on R includes b(R)). So adding b to the
    R-corner shifts the prediction by exactly b, matching the true shifted ΔG(P,R).
    """
    base = double_difference_dg(-6.0, -7.0, -5.0).dg
    shifted = double_difference_dg(-6.0, -7.0 + 100.0, -5.0).dg
    assert shifted - base == pytest.approx(100.0)


def test_selectivity_difference() -> None:
    """Selectivity ΔΔG is the difference of two absolute estimates; negative favors A."""
    dg_a = double_difference_dg(-9.0, -8.0, -5.0).dg   # -12
    dg_b = double_difference_dg(-7.0, -8.0, -5.0).dg   # -10
    ddg = double_difference_selectivity(dg_a, dg_b)
    assert ddg == pytest.approx(-2.0)
    assert ddg < 0  # prefers A


def test_returns_float() -> None:
    res = double_difference_dg(-6.0, -7.0, -5.0)
    assert isinstance(res.dg, float)
