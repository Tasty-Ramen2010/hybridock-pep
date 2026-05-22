"""Tests for MM-GBSA refinement module (scoring/mmgbsa.py).

Tests cover:
- Top-K representative selection logic (pure Python, no OpenMM)
- mmgbsa_dg mutation in refine_topk_poses (OpenMM mocked)
- CPU fallback on CUDA context failure
- best_pose.pdb MM-GBSA winner preference in csv_writer
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_poses(tmp_path: Path, n: int = 6):
    from hybridock_pep.models import ScoredPose

    poses = []
    for i in range(n):
        p = ScoredPose(
            pose_idx=i,
            pdb_path=tmp_path / f"pose_{i}.pdb",
            sequence="ACDEF",
            ca_coords=np.zeros((5, 3), dtype=np.float64),
            pdbqt_path=tmp_path / f"pose_{i}.pdbqt",
        )
        p.hybrid_score = float(-10 + i)  # -10, -9, ..., -5
        p.cluster_id = i % 3              # clusters 0, 1, 2
        poses.append(p)
    return poses


def _make_cluster_result(poses):
    from hybridock_pep.analysis.clustering import ClusterResult

    stats = []
    for cid in range(3):
        cluster_poses = [p for p in poses if p.cluster_id == cid]
        scores = [p.hybrid_score for p in cluster_poses]
        stats.append({
            "cluster_id": cid,
            "n_poses": len(cluster_poses),
            "mean_hybrid_score": float(np.mean(scores)),
            "best_pose_idx": min(cluster_poses, key=lambda p: p.hybrid_score).pose_idx,
        })
    return ClusterResult(k_optimal=3, silhouette_score=0.75, per_cluster_stats=stats)


# ---------------------------------------------------------------------------
# _select_topk_representatives
# ---------------------------------------------------------------------------

class TestSelectTopkRepresentatives:
    """Tests for _select_topk_representatives() — pure Python, no OpenMM."""

    def test_returns_one_per_cluster(self, tmp_path: Path) -> None:
        from hybridock_pep.scoring.mmgbsa import _select_topk_representatives

        poses = _make_poses(tmp_path, n=6)
        result = _make_cluster_result(poses)
        reps = _select_topk_representatives(poses, result, k=10)

        cluster_ids = [p.cluster_id for p in reps]
        assert len(cluster_ids) == len(set(cluster_ids)), "Must be one pose per cluster"

    def test_k_limits_output(self, tmp_path: Path) -> None:
        from hybridock_pep.scoring.mmgbsa import _select_topk_representatives

        poses = _make_poses(tmp_path, n=6)
        result = _make_cluster_result(poses)
        reps = _select_topk_representatives(poses, result, k=2)

        assert len(reps) == 2

    def test_best_pose_per_cluster_selected(self, tmp_path: Path) -> None:
        """Within each cluster, the pose with the lowest hybrid_score is picked."""
        from hybridock_pep.scoring.mmgbsa import _select_topk_representatives

        poses = _make_poses(tmp_path, n=6)
        result = _make_cluster_result(poses)
        reps = _select_topk_representatives(poses, result, k=10)

        for rep in reps:
            cluster_peers = [p for p in poses if p.cluster_id == rep.cluster_id]
            best_score = min(p.hybrid_score for p in cluster_peers)
            assert rep.hybrid_score == best_score, (
                f"Cluster {rep.cluster_id}: rep score {rep.hybrid_score} != best {best_score}"
            )

    def test_sorted_by_cluster_mean(self, tmp_path: Path) -> None:
        """Representatives are returned with best-mean-cluster first."""
        from hybridock_pep.scoring.mmgbsa import _select_topk_representatives

        poses = _make_poses(tmp_path, n=6)
        result = _make_cluster_result(poses)
        reps = _select_topk_representatives(poses, result, k=10)

        means = []
        for rep in reps:
            stats = next(s for s in result.per_cluster_stats if s["cluster_id"] == rep.cluster_id)
            means.append(stats["mean_hybrid_score"])
        assert means == sorted(means), "Reps must be sorted by cluster mean ascending"

    def test_no_cluster_ids_returns_empty(self, tmp_path: Path) -> None:
        from hybridock_pep.scoring.mmgbsa import _select_topk_representatives
        from hybridock_pep.analysis.clustering import ClusterResult

        poses = _make_poses(tmp_path, n=4)
        for p in poses:
            p.cluster_id = None
        result = ClusterResult(k_optimal=2, silhouette_score=0.5, per_cluster_stats=[])
        reps = _select_topk_representatives(poses, result, k=10)
        assert reps == []


# ---------------------------------------------------------------------------
# compute_mmgbsa_single — mocked OpenMM
# ---------------------------------------------------------------------------

class TestComputeMmgbsaSingle:
    """Tests for compute_mmgbsa_single() with OpenMM fully mocked."""

    def _make_pdb(self, tmp_path: Path, name: str = "pose.pdb") -> Path:
        p = tmp_path / name
        p.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
        return p

    def test_returns_float(self, tmp_path: Path) -> None:
        from hybridock_pep.scoring.mmgbsa import compute_mmgbsa_single

        pose = self._make_pdb(tmp_path, "pose.pdb")
        receptor = self._make_pdb(tmp_path, "rec.pdb")

        mock_unit = MagicMock()
        mock_unit.kilojoule_per_mole = MagicMock()
        mock_unit.kelvin = MagicMock()
        mock_unit.picosecond = MagicMock()
        mock_unit.picoseconds = MagicMock()

        # Each energy context returns a different energy (complex, receptor, peptide)
        energies = iter([-5000.0, -4000.0, -800.0])  # ΔG = (-5000+4000+800)*KJ_TO_KCAL

        def fake_potential():
            m = MagicMock()
            m.value_in_unit.return_value = next(energies)
            return m

        mock_state = MagicMock()
        mock_state.getPotentialEnergy.side_effect = fake_potential
        mock_state.getPositions.return_value = MagicMock()

        mock_ctx = MagicMock()
        mock_ctx.getState.return_value = mock_state

        mock_openmm = MagicMock()
        mock_openmm.Context.return_value = mock_ctx
        mock_openmm.Platform.getPlatformByName.return_value = MagicMock()
        mock_openmm.LocalEnergyMinimizer = MagicMock()

        mock_modeller = MagicMock()
        mock_modeller.topology = MagicMock()
        mock_modeller.positions = MagicMock()
        chains = [MagicMock(), MagicMock(), MagicMock()]
        mock_modeller.topology.chains.return_value = iter(chains)

        mock_app = MagicMock()
        mock_app.PDBFile.return_value = MagicMock(
            topology=MagicMock(chains=lambda: iter([MagicMock()])),
            positions=MagicMock(),
        )
        mock_app.Modeller.return_value = mock_modeller
        mock_app.ForceField.return_value = MagicMock()
        mock_app.NoCutoff = MagicMock()
        mock_app.HBonds = MagicMock()

        with patch.dict("sys.modules", {"openmm": mock_openmm, "openmm.app": mock_app, "openmm.unit": mock_unit}):
            with patch("hybridock_pep.scoring.mmgbsa._context_energy_kcal") as mock_energy:
                mock_ctx_obj = MagicMock()
                mock_ctx_obj.getState.return_value.getPositions.return_value = MagicMock()
                mock_energy.side_effect = [
                    (-5000.0 * 0.239006, mock_ctx_obj),   # complex
                    (-4000.0 * 0.239006, mock_ctx_obj),   # receptor
                    (-800.0 * 0.239006, mock_ctx_obj),    # peptide
                ]
                result = compute_mmgbsa_single(pose, receptor, force_cpu=True)

        assert isinstance(result, float)
        expected = (-5000.0 - (-4000.0) - (-800.0)) * 0.239006
        assert result == pytest.approx(expected, abs=0.01)

    def test_missing_openmm_raises_runtime_error(self, tmp_path: Path) -> None:
        from hybridock_pep.scoring.mmgbsa import compute_mmgbsa_single

        pose = self._make_pdb(tmp_path, "pose.pdb")
        receptor = self._make_pdb(tmp_path, "rec.pdb")

        with patch.dict("sys.modules", {"openmm": None, "openmm.app": None, "openmm.unit": None}):
            with patch("builtins.__import__", side_effect=ImportError("no openmm")):
                with pytest.raises((RuntimeError, ImportError)):
                    compute_mmgbsa_single(pose, receptor)


# ---------------------------------------------------------------------------
# refine_topk_poses — integration of selection + compute
# ---------------------------------------------------------------------------

class TestRefineTopkPoses:
    """Tests for refine_topk_poses() with compute_mmgbsa_single mocked."""

    def test_mutates_mmgbsa_dg_in_place(self, tmp_path: Path) -> None:
        from hybridock_pep.scoring.mmgbsa import refine_topk_poses
        from hybridock_pep.models import DockConfig

        poses = _make_poses(tmp_path, n=6)
        result = _make_cluster_result(poses)

        config = DockConfig(
            peptide_sequence="ACDEF",
            receptor_path=FIXTURES_DIR / "receptor_tiny.pdb",
            site_coords=(0.0, 0.0, 0.0),
            box_size=20.0,
            output_dir=tmp_path / "out",
            n_samples=10,
            refine_topk=3,
            mmgbsa_cpu_only=True,
        )

        with patch("hybridock_pep.scoring.mmgbsa.compute_mmgbsa_single", return_value=-8.5):
            refine_topk_poses(poses, result, config)

        refined = [p for p in poses if p.mmgbsa_dg is not None]
        assert len(refined) == 3
        assert all(p.mmgbsa_dg == pytest.approx(-8.5) for p in refined)

    def test_failure_per_pose_leaves_none(self, tmp_path: Path) -> None:
        """A per-pose exception must not abort the batch; mmgbsa_dg stays None."""
        from hybridock_pep.scoring.mmgbsa import refine_topk_poses
        from hybridock_pep.models import DockConfig

        poses = _make_poses(tmp_path, n=6)
        result = _make_cluster_result(poses)

        config = DockConfig(
            peptide_sequence="ACDEF",
            receptor_path=FIXTURES_DIR / "receptor_tiny.pdb",
            site_coords=(0.0, 0.0, 0.0),
            box_size=20.0,
            output_dir=tmp_path / "out",
            n_samples=10,
            refine_topk=3,
            mmgbsa_cpu_only=True,
        )

        with patch(
            "hybridock_pep.scoring.mmgbsa.compute_mmgbsa_single",
            side_effect=RuntimeError("OpenMM exploded"),
        ):
            refine_topk_poses(poses, result, config)  # must not raise

        assert all(p.mmgbsa_dg is None for p in poses)

    def test_refine_topk_none_is_noop(self, tmp_path: Path) -> None:
        from hybridock_pep.scoring.mmgbsa import refine_topk_poses
        from hybridock_pep.models import DockConfig

        poses = _make_poses(tmp_path, n=4)
        result = _make_cluster_result(poses[:3])

        config = DockConfig(
            peptide_sequence="ACDEF",
            receptor_path=FIXTURES_DIR / "receptor_tiny.pdb",
            site_coords=(0.0, 0.0, 0.0),
            box_size=20.0,
            output_dir=tmp_path / "out",
            n_samples=10,
            refine_topk=None,  # no refinement
        )

        with patch("hybridock_pep.scoring.mmgbsa.compute_mmgbsa_single") as mock_fn:
            refine_topk_poses(poses, result, config)
            mock_fn.assert_not_called()


# ---------------------------------------------------------------------------
# csv_writer: MM-GBSA winner preference in write_best_pose_pdb
# ---------------------------------------------------------------------------

class TestBestPoseMmgbsaPreference:
    """write_best_pose_pdb should prefer the MM-GBSA winner over cluster centroid."""

    def test_mmgbsa_winner_preferred(self, tmp_path: Path) -> None:
        from hybridock_pep.output.csv_writer import write_best_pose_pdb
        from hybridock_pep.analysis.clustering import ClusterResult
        from hybridock_pep.models import DockConfig, ScoredPose

        poses = []
        for i in range(3):
            p = ScoredPose(
                pose_idx=i,
                pdb_path=tmp_path / f"pose_{i}.pdb",
                sequence="ACDEF",
                ca_coords=np.zeros((5, 3)),
                pdbqt_path=tmp_path / f"pose_{i}.pdbqt",
            )
            p.hybrid_score = float(-10 + i)
            p.cluster_id = i
            (tmp_path / f"pose_{i}.pdb").write_text(f"REMARK pose {i}\n")
            poses.append(p)

        # pose 2 (worst hybrid) has the best MM-GBSA score
        poses[2].mmgbsa_dg = -15.0
        poses[0].mmgbsa_dg = -5.0

        result = ClusterResult(
            k_optimal=3,
            silhouette_score=0.7,
            per_cluster_stats=[
                {"cluster_id": 0, "mean_hybrid_score": -10.0, "best_pose_idx": 0, "n_poses": 1},
                {"cluster_id": 1, "mean_hybrid_score": -9.0, "best_pose_idx": 1, "n_poses": 1},
                {"cluster_id": 2, "mean_hybrid_score": -8.0, "best_pose_idx": 2, "n_poses": 1},
            ],
        )

        config = DockConfig(
            peptide_sequence="ACDEF",
            receptor_path=FIXTURES_DIR / "receptor_tiny.pdb",
            site_coords=(0.0, 0.0, 0.0),
            box_size=20.0,
            output_dir=tmp_path / "out",
            n_samples=3,
        )

        dest = write_best_pose_pdb(result, config, poses)
        # pose 2 has mmgbsa_dg=-15.0 (best), must win
        assert dest.read_text().strip() == "REMARK pose 2"

    def test_fallback_to_cluster_when_no_mmgbsa(self, tmp_path: Path) -> None:
        from hybridock_pep.output.csv_writer import write_best_pose_pdb
        from hybridock_pep.analysis.clustering import ClusterResult
        from hybridock_pep.models import DockConfig, ScoredPose

        poses = []
        for i in range(2):
            p = ScoredPose(
                pose_idx=i,
                pdb_path=tmp_path / f"pose_{i}.pdb",
                sequence="ACDEF",
                ca_coords=np.zeros((5, 3)),
                pdbqt_path=tmp_path / f"pose_{i}.pdbqt",
            )
            p.hybrid_score = float(-10 + i)
            p.cluster_id = i
            (tmp_path / f"pose_{i}.pdb").write_text(f"REMARK pose {i}\n")
            poses.append(p)

        result = ClusterResult(
            k_optimal=2,
            silhouette_score=0.6,
            per_cluster_stats=[
                {"cluster_id": 0, "mean_hybrid_score": -10.0, "best_pose_idx": 0, "n_poses": 1},
                {"cluster_id": 1, "mean_hybrid_score": -9.0, "best_pose_idx": 1, "n_poses": 1},
            ],
        )

        config = DockConfig(
            peptide_sequence="ACDEF",
            receptor_path=FIXTURES_DIR / "receptor_tiny.pdb",
            site_coords=(0.0, 0.0, 0.0),
            box_size=20.0,
            output_dir=tmp_path / "out",
            n_samples=2,
        )

        dest = write_best_pose_pdb(result, config, poses)
        # No mmgbsa_dg set → fallback to cluster centroid (pose 0, best mean)
        assert dest.read_text().strip() == "REMARK pose 0"
