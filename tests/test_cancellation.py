"""Tests for the two-way cancellation estimator.

The properties that matter are algebraic, so they are tested on constructed grids where the true
main effects and interaction are known, rather than on real scores where they are not.
"""
from __future__ import annotations

import numpy as np
import pytest

from hybridock_pep.scoring.cancellation import (cross_scorer, mean_centre, median_polish,
                                                two_way)


def _grid(n=6, m=7, seed=0):
    """A grid built from known main effects plus a known interaction."""
    rng = np.random.default_rng(seed)
    row = rng.normal(0, 3, n)
    col = rng.normal(0, 5, m)
    inter = rng.normal(0, 1, (n, m))
    return 10.0 + row[:, None] + col[None, :] + inter, row, col, inter


def test_mean_centre_removes_pure_main_effects() -> None:
    """A grid with NO interaction must centre to zero."""
    row = np.array([1.0, -2.0, 4.0])
    col = np.array([0.5, 3.0, -1.0, 2.0])
    M = 7.0 + row[:, None] + col[None, :]
    assert np.allclose(mean_centre(M), 0.0, atol=1e-12)


def test_median_polish_removes_pure_main_effects() -> None:
    row = np.array([1.0, -2.0, 4.0, 0.0])
    col = np.array([0.5, 3.0, -1.0])
    M = -4.0 + row[:, None] + col[None, :]
    resid, _, _, _ = median_polish(M)
    assert np.abs(resid).max() < 1e-9


def test_median_polish_reconstructs_the_table() -> None:
    M, _, _, _ = _grid()
    resid, row, col, grand = median_polish(M)
    assert np.allclose(resid + row[:, None] + col[None, :] + grand, M, atol=1e-9)


def test_median_polish_recovers_the_interaction() -> None:
    M, _, _, inter = _grid()
    resid = median_polish(M)[0]
    # the recovered interaction differs from the truth by its own main effects, which is exactly
    # what is unidentifiable in a two-way model -- so compare the doubly-centred versions
    assert np.corrcoef(mean_centre(resid).ravel(), mean_centre(inter).ravel())[0, 1] > 0.99


def test_median_polish_survives_outliers_where_the_mean_does_not() -> None:
    """The point of the robust version: a few blown-up cells must not move the EFFECTS.

    A docked grid always contains clashed poses worth hundreds of REU. A mean sweeps them into
    the "main effect" it removes, so every OTHER cell in that row and column is shifted; a median
    leaves them alone. The comparison is therefore on the estimated effects and on the clean
    cells -- the corrupted cells themselves are wrong under either estimator, and including them
    hides the difference entirely.
    """
    M, row, col, inter = _grid(seed=3)
    dirty = M.copy()
    bad = [(1, 2), (4, 5)]
    for i, j in bad:
        dirty[i, j] += 900.0

    _, row_med, col_med, _ = median_polish(dirty)
    row_mean = dirty.mean(1) - dirty.mean()
    col_mean = dirty.mean(0) - dirty.mean()
    # effects are identified only up to a shift, so compare them centred
    err_med = np.abs((row_med - row_med.mean()) - (row - row.mean())).max()
    err_mean = np.abs((row_mean - row_mean.mean()) - (row - row.mean())).max()
    assert err_med < err_mean, f"median {err_med:.2f} should beat mean {err_mean:.2f}"
    assert err_med < 1.0, "median polish should barely notice two bad cells"
    assert err_mean > 50.0, "the mean should be badly dragged by a +900 cell"

    # and on the CLEAN cells the recovered interaction should track the truth better.
    # Compare after removing each estimate's own MEDIAN, not its mean: re-mean-centring a
    # residual that still contains the two +900 cells simply re-contaminates it, which made an
    # earlier version of this test report the two methods as exactly equal.
    keep = np.ones_like(dirty, dtype=bool)
    for i, j in bad:
        keep[i, j] = False
    truth = (inter - np.median(inter))[keep]
    r_med = median_polish(dirty)[0]
    r_mn = mean_centre(dirty)
    c_med = np.corrcoef((r_med - np.median(r_med))[keep], truth)[0, 1]
    c_mean = np.corrcoef((r_mn - np.median(r_mn))[keep], truth)[0, 1]
    assert c_med > c_mean, f"median {c_med:.3f} should beat mean {c_mean:.3f}"
    assert c_med > 0.85


def test_two_way_scale_balancing_equalises_row_spread() -> None:
    """A row that merely swings harder must not end up with larger residuals for free."""
    rng = np.random.default_rng(1)
    M = rng.normal(0, 1, (6, 6))
    M[0] *= 8.0                       # one peptide with a much wider range
    R = two_way(M, robust=True, scale=True)
    spreads = np.median(np.abs(R - np.median(R, axis=1, keepdims=True)), axis=1)
    assert spreads.max() / max(spreads.min(), 1e-9) < 3.0


def test_two_way_rejects_non_matrix() -> None:
    with pytest.raises(ValueError, match="2-D grid"):
        two_way(np.zeros(5))


def test_cross_scorer_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        cross_scorer(np.zeros((3, 3)), np.zeros((3, 4)))


def test_cross_scorer_zero_weight_is_plain_cancellation() -> None:
    A, _, _, _ = _grid(seed=5)
    B, _, _, _ = _grid(seed=6)
    assert np.allclose(cross_scorer(A, B, weight=0.0), two_way(A, robust=True), atol=1e-9)


def test_cross_scorer_injects_the_donors_effects() -> None:
    """With weight > 0 the donor's per-column effect must show up in the result."""
    A, _, _, _ = _grid(n=6, m=6, seed=7)
    col = np.array([5.0, -5.0, 0.0, 2.0, -2.0, 1.0])
    B = col[None, :] + np.zeros((6, 6))          # donor carries ONLY a column effect
    out = cross_scorer(A, B, weight=1.0)
    delta = out - two_way(A, robust=True)
    # every row should have been shifted by the same column pattern
    assert np.corrcoef(delta.mean(0), col)[0, 1] > 0.99
