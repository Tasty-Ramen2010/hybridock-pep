"""Tests for within-target charge complementarity (selectivity ranking, not absolute ΔG)."""
from __future__ import annotations

import numpy as np

from hybridock_pep.scoring.charge_complementarity import charge_complementarity_score


def test_salt_bridge_is_favorable():
    """A Lys near an Asp (opposite charges, 4 Å) must give a negative salt_bridge term."""
    pep = [("K", np.array([0.0, 0.0, 0.0]))]
    rec = [("D", np.array([4.0, 0.0, 0.0]))]
    s = charge_complementarity_score(pep, rec)
    assert s["salt_bridge"] < 0
    assert s["n_salt_bridges"] == 1
    assert s["repulsion"] == 0.0


def test_like_charge_is_repulsive():
    """Lys near Arg (like charges) → positive repulsion, no salt bridge."""
    pep = [("K", np.array([0.0, 0.0, 0.0]))]
    rec = [("R", np.array([4.0, 0.0, 0.0]))]
    s = charge_complementarity_score(pep, rec)
    assert s["repulsion"] > 0
    assert s["n_repulsive"] == 1
    assert s["salt_bridge"] == 0.0


def test_distance_cutoff_and_screening():
    """Beyond cutoff there is no interaction; closer pairs are stronger."""
    pep = [("K", np.array([0.0, 0.0, 0.0]))]
    far = charge_complementarity_score(pep, [("D", np.array([20.0, 0.0, 0.0]))])
    assert far["net_elec"] == 0.0
    near = charge_complementarity_score(pep, [("D", np.array([3.0, 0.0, 0.0]))])
    mid = charge_complementarity_score(pep, [("D", np.array([5.0, 0.0, 0.0]))])
    assert abs(near["salt_bridge"]) > abs(mid["salt_bridge"])


def test_neutral_residues_ignored():
    """Non-charged residues contribute nothing."""
    s = charge_complementarity_score([("A", np.zeros(3)), ("L", np.ones(3))],
                                     [("G", np.array([3.0, 0.0, 0.0]))])
    assert s["net_elec"] == 0.0
