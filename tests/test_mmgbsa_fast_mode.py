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

from hybridock_pep.scoring import mmgbsa


def _app():
    """Import openmm.app lazily.

    Deliberately NOT imported at module scope: doing so puts the real
    ``openmm.app`` into ``sys.modules`` at collection time, which defeats the
    MagicMock-based openmm patching in tests/test_mmgbsa.py and makes
    TestComputeMmgbsaSingle::test_returns_float fail with "Structure contains
    no atoms". Collection-time imports are shared state.
    """
    return pytest.importorskip("openmm.app", reason="OpenMM not installed")


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
    assert mmgbsa._minimize_constraints() is _app().HBonds


def test_fast_mode_drops_constraints():
    os.environ["HYBRIDOCK_MMGBSA_FAST"] = "1"
    assert mmgbsa._minimize_constraints() is None


@pytest.mark.parametrize("value", ["0", "", "true", "yes", "TRUE", "2", "garbage"])
def test_only_exact_1_enables_fast_mode(value):
    """Anything but a literal "1" keeps the safe default — no truthiness games,
    so a stray value can never silently change scoring."""
    os.environ["HYBRIDOCK_MMGBSA_FAST"] = value
    assert mmgbsa._minimize_constraints() is _app().HBonds


def test_unset_env_is_default():
    assert "HYBRIDOCK_MMGBSA_FAST" not in os.environ
    assert mmgbsa._minimize_constraints() is _app().HBonds


# ---------------------------------------------------------------------------
# Residual-force convergence guard
# ---------------------------------------------------------------------------

class _FakeState:
    def __init__(self, forces):
        self._f = forces

    def getForces(self):
        class _Q:
            def __init__(self, f):
                self._f = f

            def value_in_unit(self, _unit):
                return self._f
        return _Q(self._f)


class _FakeCtx:
    def __init__(self, forces):
        self._f = forces

    def getState(self, **_kw):
        return _FakeState(self._f)


def test_converged_minimization_does_not_warn(caplog):
    """Real converged runs measured 54-102 kJ/mol/nm; these must stay silent."""
    ctx = _FakeCtx([(60.0, 0.0, 0.0), (0.0, 80.0, 0.0)])
    with caplog.at_level("WARNING"):
        max_f = mmgbsa._warn_if_unconverged(ctx)
    assert max_f == pytest.approx(80.0)
    assert not caplog.records


def test_diverged_minimization_warns(caplog):
    """A single-precision basin-hop must be surfaced, not silently ranked."""
    ctx = _FakeCtx([(0.0, 0.0, 5000.0)])
    with caplog.at_level("WARNING"):
        max_f = mmgbsa._warn_if_unconverged(ctx)
    assert max_f == pytest.approx(5000.0)
    assert any("did not converge" in r.message for r in caplog.records)


def test_guard_never_raises_on_bad_context():
    """A diagnostic must not be able to break scoring."""
    class Broken:
        def getState(self, **_kw):
            raise RuntimeError("boom")

    import math
    assert math.isnan(mmgbsa._warn_if_unconverged(Broken()))
