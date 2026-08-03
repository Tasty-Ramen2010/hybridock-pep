"""Blind docking: when no pocket is known, the off-pocket filter must be off.

Stage 1.7a drops any pose whose Cα centroid is more than a fixed 35 Å from
``site_coords``. That is correct when the caller pointed at a real pocket. It is
actively wrong when they could not: with ``site_coords`` set to the receptor's
own centroid, the cutoff is a 35 Å sphere around the middle of the protein, so
on any receptor whose half-extent exceeds 35 Å it cannot reach its own surface.
Poses sitting in a genuine groove at either end get deleted for being far from a
point that never meant anything.

The damage scales with receptor size, which is the part that makes results
non-comparable: on a ProP-PD specificity benchmark, 206 of 1,185 pairs sat on
receptors (armadillo and ankyrin repeats, coiled coils, an Alix V-domain) whose
half-extent was 35–44 Å, so those receptors were silently scored on a truncated
pose set while compact ones were not.
"""

from __future__ import annotations

import numpy as np
import pytest

from hybridock_pep.driver import _OFFPOCKET_CENTROID_AA, _filter_offpocket_poses
from hybridock_pep.models import DockConfig


def _receptor(tmp_path):
    """DockConfig validates that receptor_path exists, so write a real one."""
    p = tmp_path / "r.pdb"
    p.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    return p


@pytest.fixture()
def config(tmp_path):
    return DockConfig(
        peptide_sequence="ACDEFGHIK",
        receptor_path=_receptor(tmp_path),
        site_coords=(0.0, 0.0, 0.0),
        box_size=100.0,
        output_dir=tmp_path / "out",
    )


class _Rec:
    """Minimal stand-in for PoseRecord: the filter reads ca_coords and pose_idx."""

    def __init__(self, idx, centroid):
        self.pose_idx = idx
        self.ca_coords = np.array([centroid], dtype=float)


class TestBlindFlagDefaults:
    def test_blind_is_off_by_default(self, config):
        assert config.blind is False, (
            "blind must be opt-in — turning the off-pocket filter off by default "
            "would change every existing pocket-directed run"
        )

    def test_blind_is_settable(self, tmp_path):
        c = DockConfig(
            peptide_sequence="ACDEFGHIK",
            receptor_path=_receptor(tmp_path),
            site_coords=(0.0, 0.0, 0.0),
            box_size=100.0,
            output_dir=tmp_path / "out",
            blind=True,
        )
        assert c.blind is True


class TestFilterReachability:
    """The geometry that makes this a real bug, not a preference."""

    def test_filter_cannot_reach_past_its_radius(self, config):
        """A pose on the far surface of an elongated receptor is dropped."""
        near = _Rec(0, (0.0, 0.0, 0.0))
        far = _Rec(1, (0.0, 0.0, _OFFPOCKET_CENTROID_AA + 9.0))  # 44 Å, KPNA4-scale

        kept = _filter_offpocket_poses(config, [near, far])

        assert [r.pose_idx for r in kept] == [0], (
            "a pose 44 Å from the centroid survived; the fixture no longer "
            "reproduces the truncation this test exists for"
        )

    def test_truncation_is_receptor_size_dependent(self, config):
        """The same pose survives on a compact receptor and dies on a long one.

        This asymmetry is what makes cross-receptor comparison invalid: the
        filter removes more of the pose set the bigger the receptor gets.
        """
        surface_at = lambda d: _Rec(0, (0.0, 0.0, d))  # noqa: E731

        compact_kept = _filter_offpocket_poses(config, [surface_at(20.0)])
        elongated_kept = _filter_offpocket_poses(config, [surface_at(44.0)])

        assert len(compact_kept) == 1
        assert len(elongated_kept) == 0


class TestDriverSkipsFilterWhenBlind:
    """The filter call itself must be bypassed, not merely widened."""

    def test_blind_run_keeps_every_pose(self, monkeypatch, tmp_path):
        import hybridock_pep.driver as driver_mod

        called = []
        monkeypatch.setattr(
            driver_mod, "_filter_offpocket_poses",
            lambda cfg, recs, log=None: called.append(True) or recs,
        )

        # Exercise the branch as driver.run_dock has it, without standing up a
        # full pipeline run.
        records = [_Rec(0, (0.0, 0.0, 0.0)), _Rec(1, (0.0, 0.0, 90.0))]
        for blind in (True, False):
            cfg = DockConfig(
                peptide_sequence="ACDEFGHIK",
                receptor_path=_receptor(tmp_path),
                site_coords=(0.0, 0.0, 0.0),
                box_size=100.0,
                output_dir=tmp_path / "out",
                blind=blind,
            )
            if cfg.blind:
                kept = records
            else:
                kept = driver_mod._filter_offpocket_poses(cfg, records)
            assert len(kept) == 2

        assert called == [True], (
            "the filter should be invoked exactly once — skipped under blind, "
            "run otherwise"
        )

    def test_source_guards_the_filter_on_blind(self):
        """Guard against the branch being refactored away."""
        from pathlib import Path

        src = Path(driver_source()).read_text()
        idx = src.find("Stage 1.7a")
        assert idx != -1
        window = src[idx: idx + 1600]
        assert "config.blind" in window, (
            "Stage 1.7a no longer checks config.blind — blind runs are back to "
            "having their off-pocket poses deleted"
        )


def driver_source() -> str:
    import hybridock_pep.driver as driver_mod

    return driver_mod.__file__


class TestCliWiring:
    def test_blind_flag_exists_and_relaxes_site_box(self):
        """--site/--box become optional under --blind, and stay required without."""
        from hybridock_pep.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "dock", "--peptide", "ACDEFGHIK", "--receptor", "r.pdb", "--blind",
            "--output-dir", "out",
        ])
        assert args.blind is True
        assert args.site is None and args.box is None

    def test_blind_still_accepts_an_explicit_box(self):
        """A caller who knows the receptor's extent should be able to bound the
        Stage 2 grid without pretending to know a pocket."""
        from hybridock_pep.cli import _build_parser

        args = _build_parser().parse_args([
            "dock", "--peptide", "ACDEFGHIK", "--receptor", "r.pdb", "--blind",
            "--site", "1", "2", "3", "--box", "60", "--output-dir", "out",
        ])
        assert args.blind is True
        assert args.site == [1.0, 2.0, 3.0]
        assert args.box == 60.0

    def test_receptor_centroid_helper(self, tmp_path):
        from hybridock_pep.cli import _receptor_geometric_center

        pdb = tmp_path / "r.pdb"
        pdb.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      2  CA  ALA A   2      10.000  20.000  30.000  1.00  0.00           C\n"
            "END\n"
        )
        assert _receptor_geometric_center(pdb) == (5.0, 10.0, 15.0)

    def test_centroid_helper_rejects_an_empty_pdb(self, tmp_path):
        from hybridock_pep.cli import _receptor_geometric_center

        pdb = tmp_path / "empty.pdb"
        pdb.write_text("REMARK nothing here\nEND\n")
        with pytest.raises(ValueError, match="No ATOM/HETATM"):
            _receptor_geometric_center(pdb)
