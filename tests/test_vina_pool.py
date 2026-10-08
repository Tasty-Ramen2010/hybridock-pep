"""Tests for the memory-aware Vina worker pool and the shared per-pose scorer."""
from __future__ import annotations

import logging
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from hybridock_pep.models import DockConfig, ScoredPose
from hybridock_pep.scoring import vina as vmod

_SITE = (0.0, 0.0, 0.0)
_BOX = 20.0
_ATOM = "ATOM      1  C   ALA A   1    {x:8.3f}   0.000   0.000  1.00  0.00    +0.000 C \n"


def _pose(tmp: Path, idx: int, x: float = 1.0) -> ScoredPose:
    f = tmp / f"pose_{idx}.pdbqt"
    f.write_text(_ATOM.format(x=x))
    return ScoredPose(pose_idx=idx, pdb_path=tmp / f"pose_{idx}.pdb", sequence="A",
                      ca_coords=np.zeros((1, 3)), pdbqt_path=f)


def _cfg(tmp: Path) -> DockConfig:
    rec = tmp / "r.pdbqt"
    rec.write_text("REMARK\n")
    return DockConfig(peptide_sequence="A", receptor_path=rec, site_coords=_SITE,
                      box_size=_BOX, output_dir=tmp)


class TestWorkerCount:
    def test_small_batches_stay_in_process(self) -> None:
        assert vmod._vina_worker_count(3, 30.0, 164) == 0

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HYBRIDOCK_VINA_WORKERS", "3")
        assert vmod._vina_worker_count(100, 30.0, 164) == 3
        assert vmod._vina_worker_count(2, 30.0, 164) == 0  # batch-size floor still applies

    def test_tight_budget_means_in_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HYBRIDOCK_VINA_WORKERS", raising=False)
        with mock.patch("hybridock_pep.hardware.memory_budget_bytes", return_value=10 * 1024**3):
            assert vmod._vina_worker_count(100, 60.0, 164) == 0  # needs ~15 GB per worker

    def test_blind_box_does_not_fit_24g(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HYBRIDOCK_VINA_WORKERS", raising=False)
        with mock.patch("hybridock_pep.hardware.memory_budget_bytes", return_value=21 * 1024**3):
            assert vmod._vina_worker_count(450, 100.0, 164) == 0

    def test_ample_budget_scales_with_memory_not_cores(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HYBRIDOCK_VINA_WORKERS", raising=False)
        with mock.patch("hybridock_pep.hardware.memory_budget_bytes", return_value=100 * 1024**3), \
                mock.patch("hybridock_pep.hardware.cpu_threads", return_value=12):
            n = vmod._vina_worker_count(100, 60.0, 164)
        assert 5 <= n <= 7  # ~0.9*100 GB / ~15 GB

    def test_memory_cost_is_quadratic_in_ligand_atoms(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HYBRIDOCK_VINA_WORKERS", raising=False)
        with mock.patch("hybridock_pep.hardware.memory_budget_bytes", return_value=40 * 1024**3), \
                mock.patch("hybridock_pep.hardware.cpu_threads", return_value=24):
            small = vmod._vina_worker_count(200, 30.0, 80)
            big = vmod._vina_worker_count(200, 30.0, 164)
        assert small > 2 * big


class TestScoreOnePose:
    def test_clash_relieved(self, tmp_path: Path) -> None:
        v = mock.MagicMock()
        v.score.side_effect = [[50.0], [-3.0]]
        p = _pose(tmp_path, 0)
        status, score, clipped, err, _ = vmod._score_one_pose(
            v, 0, p.pdbqt_path, _SITE, _BOX, True, 5)
        assert (status, score, clipped, err) == ("scored", -3.0, False, None)
        v.write_pose.assert_called_once()

    def test_clash_not_relieved_is_failure(self, tmp_path: Path) -> None:
        v = mock.MagicMock()
        v.score.side_effect = [[900.0], [800.0], [799.9]]
        p = _pose(tmp_path, 0)
        status, score, _, err, _ = vmod._score_one_pose(v, 0, p.pdbqt_path, _SITE, _BOX, True, 5)
        assert status == "failed" and score is None and "clash_relief_failed" in (err or "")
        v.write_pose.assert_not_called()

    def test_clipped_skips_vina(self, tmp_path: Path) -> None:
        v = mock.MagicMock()
        p = _pose(tmp_path, 0, x=500.0)
        status, _, clipped, err, _ = vmod._score_one_pose(v, 0, p.pdbqt_path, _SITE, _BOX, True, 5)
        assert status == "clipped" and clipped and (err or "").startswith("is_clipped:")
        v.set_ligand_from_file.assert_not_called()


class TestBatch:
    def _run(self, tmp_path: Path, n: int, workers: int):
        poses = [_pose(tmp_path, i) for i in range(n)]
        v = mock.MagicMock()
        v.score.side_effect = lambda: [-float(len(v.set_ligand_from_file.call_args_list))]
        with mock.patch.object(vmod, "Vina", return_value=v), \
                mock.patch.object(vmod, "_vina_worker_count", return_value=workers):
            return poses, vmod.score_vina_batch(poses, _cfg(tmp_path), tmp_path / "r.pdbqt")

    def test_in_process_preserves_order(self, tmp_path: Path) -> None:
        _, (scored, failures) = self._run(tmp_path, 6, 0)
        assert [p.pose_idx for p in scored] == list(range(6)) and not failures

    def test_pool_failure_falls_back_in_process(self, tmp_path: Path, caplog) -> None:
        poses = [_pose(tmp_path, i) for i in range(5)]
        v = mock.MagicMock()
        v.score.return_value = [-4.0]

        class Boom:
            def __init__(self, *a, **k):
                raise BrokenProcessPool("worker died")

        with mock.patch.object(vmod, "Vina", return_value=v), \
                mock.patch.object(vmod, "_vina_worker_count", return_value=2), \
                mock.patch.object(vmod, "ProcessPoolExecutor", Boom), \
                caplog.at_level(logging.WARNING):
            scored, failures = vmod.score_vina_batch(poses, _cfg(tmp_path), tmp_path / "r.pdbqt")
        assert len(scored) == 5 and not failures
        assert "worker pool failed" in caplog.text

    def test_malloc_env_restored_after_pool(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("MALLOC_MMAP_MAX_", raising=False)

        class Boom:
            def __init__(self, *a, **k):
                import os
                assert os.environ["MALLOC_MMAP_MAX_"] == "0"  # visible to would-be workers
                raise BrokenProcessPool("x")

        poses = [_pose(tmp_path, i) for i in range(4)]
        v = mock.MagicMock()
        v.score.return_value = [-1.0]
        with mock.patch.object(vmod, "Vina", return_value=v), \
                mock.patch.object(vmod, "_vina_worker_count", return_value=1), \
                mock.patch.object(vmod, "ProcessPoolExecutor", Boom):
            vmod.score_vina_batch(poses, _cfg(tmp_path), tmp_path / "r.pdbqt")
        import os
        assert "MALLOC_MMAP_MAX_" not in os.environ
