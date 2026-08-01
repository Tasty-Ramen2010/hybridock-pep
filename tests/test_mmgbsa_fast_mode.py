"""HYBRIDOCK_MMGBSA_FAST — the constraint switch that dominates refine runtime.

`LocalEnergyMinimizer` has no native constraint support: it converts constraints
to harmonic restraints and re-runs L-BFGS in an outer loop until they are
satisfied. Since minimisation is 94% of MM-GBSA's runtime, `constraints=HBonds`
multiplies the whole stage — measured 21.1 s/pose against 4.3 s/pose without,
a 4.9x difference on an M3 (5.41x through the public API).

It stays off by default because it moves the answer (up to ~4.5 kcal/mol, and
systematically for at least one pose), and MM-GBSA ranks poses. These tests pin
the default so it cannot drift on silently.

Fast by construction: only the constraint selector is exercised, no OpenMM
Context is ever built.
"""

import os

import pytest

app = pytest.importorskip("openmm.app", reason="OpenMM not installed")

from hybridock_pep.scoring import mmgbsa


@pytest.fixture(autouse=True)
def _clean_env():
    """Never leak the flag between tests or out into the rest of the suite."""
    orig = os.environ.get("HYBRIDOCK_MMGBSA_FAST")
    os.environ.pop("HYBRIDOCK_MMGBSA_FAST", None)
    yield
    os.environ.pop("HYBRIDOCK_MMGBSA_FAST", None)
    if orig is not None:
        os.environ["HYBRIDOCK_MMGBSA_FAST"] = orig


def test_default_is_hbonds():
    """The scientific default must not change without re-benchmarking."""
    assert mmgbsa._minimize_constraints() is app.HBonds


def test_fast_mode_drops_constraints():
    os.environ["HYBRIDOCK_MMGBSA_FAST"] = "1"
    assert mmgbsa._minimize_constraints() is None


@pytest.mark.parametrize("value", ["0", "", "true", "yes", "TRUE", "2", "garbage"])
def test_only_exact_1_enables_fast_mode(value):
    """Anything but a literal "1" keeps the safe default — no truthiness games,
    so a stray value can never silently change scoring."""
    os.environ["HYBRIDOCK_MMGBSA_FAST"] = value
    assert mmgbsa._minimize_constraints() is app.HBonds


def test_unset_env_is_default():
    assert "HYBRIDOCK_MMGBSA_FAST" not in os.environ
    assert mmgbsa._minimize_constraints() is app.HBonds
