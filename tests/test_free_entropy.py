"""Unit tests for the free-state conformational entropy feature (SCORE-ENT).

The MD itself (run_free_dynamics) requires OpenMM + GPU and is not exercised here; we test
the pure helper s_free_buried and that compute_free_state_entropy degrades gracefully when the
MD machinery is unavailable or fails.
"""
from __future__ import annotations

from pathlib import Path

from hybridock_pep.scoring.free_entropy import compute_free_state_entropy, s_free_buried


def test_s_free_buried_scales_with_burial() -> None:
    assert s_free_buried(1.0, 0.0) == 0.0
    assert s_free_buried(1.0, 1.0) == 1.0
    assert s_free_buried(0.8, 0.5) == 0.4


def test_s_free_buried_clamps_out_of_range() -> None:
    assert s_free_buried(1.0, 1.5) == 1.0   # buried fraction clamped to 1
    assert s_free_buried(1.0, -0.2) == 0.0  # clamped to 0


def test_compute_free_state_entropy_missing_file_returns_none() -> None:
    # nonexistent PDB → MD fails → None (never raises), so the ensemble stage can skip cleanly.
    result = compute_free_state_entropy(Path("/nonexistent/peptide.pdb"), prod_ps=10)
    assert result is None
