"""Tests for the co-folding stage.

The geometry tests use synthetic coordinates so they are fast and deterministic. The one
real-structure test is marked slow and skipped when the 9CCE fixtures are not present --
that pair (a Boltz-2 prediction of a complex released after its training cutoff, plus the
crystal structure) is the case the whole stage was designed around, so it is worth asserting
on when it is available.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hybridock_pep.analysis.direction import (
    MIN_AXIS_QUALITY,
    agreement,
    axis,
    axis_quality,
    filter_by_direction,
)
from hybridock_pep.sampling.cofold import kabsch

_FIX = Path(__file__).parent / "fixtures/cofold"
#: 9CCE chains B (receptor) and C (the 11-mer it binds); A pairs with D and is not used.
_XTAL = _FIX / "9cce_chainBC.pdb"
#: Boltz-2 co-fold of the same two sequences, run locally with no MSA.
_BOLTZ = _FIX / "9cce_boltz2_cofold.pdb"


def _strand(n: int = 12, spacing: float = 3.5) -> np.ndarray:
    """An idealised extended peptide along +x."""
    return np.stack([np.arange(n) * spacing, np.zeros(n), np.zeros(n)], axis=1)


def test_axis_points_n_to_c() -> None:
    assert np.allclose(axis(_strand()), [1.0, 0.0, 0.0])


def test_axis_of_reversed_strand_is_opposite() -> None:
    assert np.allclose(axis(_strand()[::-1]), [-1.0, 0.0, 0.0])


def test_axis_needs_two_residues() -> None:
    with pytest.raises(ValueError, match=">=2 residues"):
        axis(np.zeros((1, 3)))


def test_axis_quality_extended_vs_hairpin() -> None:
    extended = _strand()
    assert axis_quality(extended) > 0.99
    # a hairpin doubles back, so end-to-end distance collapses against contour length
    hairpin = np.concatenate([_strand(6), _strand(6)[::-1] + [0.0, 5.0, 0.0]])
    assert axis_quality(hairpin) < 0.3


def test_agreement_is_signed() -> None:
    fwd = _strand()
    assert agreement(fwd, axis(fwd)) == pytest.approx(1.0)
    assert agreement(fwd[::-1], axis(fwd)) == pytest.approx(-1.0)


def test_filter_keeps_only_forward_poses() -> None:
    fwd = _strand()
    poses = [(f"f{i}", fwd + i) for i in range(5)] + [(f"b{i}", fwd[::-1] + i) for i in range(5)]
    kept, diag = filter_by_direction(poses, axis(fwd), fwd)
    assert diag["applied"] is True
    assert sorted(kept) == [f"f{i}" for i in range(5)]


def test_filter_refuses_when_prior_is_not_extended() -> None:
    """A prior whose own axis is meaningless must not be allowed to delete poses."""
    blob = np.array([[0.0, 0, 0], [3.5, 0, 0], [3.5, 3.5, 0], [0.2, 0.4, 0]])
    assert axis_quality(blob) < MIN_AXIS_QUALITY
    poses = [(f"p{i}", _strand() + i) for i in range(6)]
    kept, diag = filter_by_direction(poses, axis(blob), blob)
    assert diag["applied"] is False
    assert len(kept) == len(poses)
    assert "not extended enough" in diag["reason"]


def test_filter_abandons_rather_than_leaving_too_few() -> None:
    """Replacing an ensemble with one surviving pose is worse than not filtering."""
    fwd = _strand()
    poses = [("f0", fwd)] + [(f"b{i}", fwd[::-1] + i) for i in range(9)]
    kept, diag = filter_by_direction(poses, axis(fwd), fwd, min_keep=3)
    assert diag["applied"] is False
    assert len(kept) == 10
    assert diag["n_forward"] == 1


def test_kabsch_recovers_a_known_rigid_transform() -> None:
    rng = np.random.default_rng(0)
    P = rng.normal(size=(20, 3))
    theta = 0.7
    R_true = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta), np.cos(theta), 0],
                       [0, 0, 1.0]])
    t_true = np.array([4.0, -2.0, 1.5])
    Q = (R_true @ P.T).T + t_true
    R, t, rmsd = kabsch(P, Q)
    assert rmsd == pytest.approx(0.0, abs=1e-8)
    assert np.allclose((R @ P.T).T + t, Q, atol=1e-8)


def test_kabsch_rejects_mismatched_input() -> None:
    with pytest.raises(ValueError, match=">=3 matched points"):
        kabsch(np.zeros((2, 3)), np.zeros((2, 3)))


@pytest.mark.slow
@pytest.mark.skipif(not (_XTAL.exists() and _BOLTZ.exists()),
                    reason="9CCE crystal / Boltz fixtures not present")
def test_transplant_9cce_lands_on_the_crystal_peptide() -> None:
    """The motivating case: a co-folded peptide moved onto the crystal receptor.

    9CCE was released after Boltz-2's training cutoff, so this is a genuine prediction rather
    than recall. The fixture holds chain B (the receptor) and chain C (the 11-mer it binds).
    """
    import tempfile

    from hybridock_pep.sampling.cofold import read_chains, transplant

    with tempfile.TemporaryDirectory() as td:
        rec = Path(td) / "recB.pdb"
        rec.write_text("\n".join(l for l in _XTAL.read_text().splitlines()
                                 if l.startswith("ATOM") and l[21] == "B") + "\nEND\n")
        tr = transplant(_BOLTZ, rec)

    assert tr.accepted, f"fold RMSD {tr.fold_rmsd:.2f} A should be well inside the gate"
    assert tr.fold_rmsd < 1.5
    crystal_pep = read_chains(_XTAL)["C"]
    n = min(len(tr.peptide_ca), len(crystal_pep.ca))
    direct = float(np.sqrt(((tr.peptide_ca[:n] - crystal_pep.ca[:n]) ** 2).sum(1).mean()))
    reversed_ = float(np.sqrt(
        ((tr.peptide_ca[:n] - crystal_pep.ca[:n][::-1]) ** 2).sum(1).mean()))
    assert direct < 3.0, f"transplanted peptide {direct:.2f} A from crystal"
    assert reversed_ > 10.0, "the reversed threading must be clearly distinguishable"
    assert agreement(tr.peptide_ca, axis(crystal_pep.ca)) > 0.9
