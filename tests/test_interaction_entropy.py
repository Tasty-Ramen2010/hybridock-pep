"""Unit tests for the Interaction Entropy estimator (no OpenMM required)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from hybridock_pep.scoring.interaction_entropy import (
    _KT_300,
    interaction_entropy,
)


def test_zero_fluctuation_gives_zero_entropy():
    """A perfectly rigid interface (constant E_int) has -TΔS_IE = 0."""
    e = np.full(50, -42.0)
    assert interaction_entropy(e) == pytest.approx(0.0, abs=1e-9)


def test_entropy_is_positive_and_grows_with_fluctuation():
    """Larger interaction-energy fluctuation → larger entropy penalty."""
    rng = np.random.default_rng(0)
    small = interaction_entropy(rng.normal(-30.0, 1.0, 4000))
    large = interaction_entropy(rng.normal(-30.0, 4.0, 4000))
    assert small > 0.0
    assert large > small


def test_matches_gaussian_closed_form():
    """For Gaussian ΔE, -TΔS_IE → σ²/(2kT) in the large-sample limit.

    ln⟨exp(βΔE)⟩ = β²σ²/2 for a zero-mean Gaussian, so -TΔS = kT·β²σ²/2 = σ²/(2kT).
    """
    sigma = 2.5
    rng = np.random.default_rng(1)
    e = rng.normal(0.0, sigma, 200_000)
    expected = sigma**2 / (2.0 * _KT_300)
    assert interaction_entropy(e) == pytest.approx(expected, rel=0.05)


def test_invariant_to_mean_shift():
    """IE depends only on fluctuations, not the absolute interaction energy."""
    rng = np.random.default_rng(2)
    base = rng.normal(0.0, 3.0, 5000)
    a = interaction_entropy(base)
    b = interaction_entropy(base + 1000.0)
    assert a == pytest.approx(b, rel=1e-6)


def test_requires_two_frames():
    with pytest.raises(ValueError):
        interaction_entropy([1.0])


def test_temperature_scaling_closed_form():
    """At higher T the same fluctuation yields a smaller σ²/(2kT) penalty."""
    sigma = 2.0
    rng = np.random.default_rng(3)
    e = rng.normal(0.0, sigma, 200_000)
    s300 = interaction_entropy(e, temperature_k=300.0)
    s400 = interaction_entropy(e, temperature_k=400.0)
    assert s400 < s300
    kt400 = _KT_300 * (400.0 / 300.0)
    assert s400 == pytest.approx(sigma**2 / (2.0 * kt400), rel=0.05)
