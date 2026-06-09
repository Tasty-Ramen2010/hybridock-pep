"""Tests for the two-step / gating / IE / 3-traj options in refine_topk_poses.

OpenMM is mocked via compute_mmgbsa_single, so these run without a force field.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from hybridock_pep.models import DockConfig, ScoredPose
from hybridock_pep.analysis.clustering import ClusterResult
from hybridock_pep.scoring import mmgbsa


def _pose(idx: int, hybrid: float, *, clashed: bool = False, cid: int = 0) -> ScoredPose:
    return ScoredPose(
        pose_idx=idx, pdb_path=Path(f"/tmp/p{idx}.pdb"), sequence="LIS",
        ca_coords=np.zeros((3, 3)), hybrid_score=hybrid, cluster_id=cid,
        is_clashed=clashed,
    )


def _cfg(tmp_path: Path, **kw) -> DockConfig:
    return DockConfig(
        peptide_sequence="LIS", receptor_path=tmp_path / "r.pdb",
        site_coords=(0.0, 0.0, 0.0), box_size=20.0, output_dir=tmp_path,
        refine_topk=5, mmgbsa_cpu_only=True, **kw,
    )


def _cluster(cids: list[int]) -> ClusterResult:
    stats = [{"cluster_id": c, "mean_hybrid_score": -10.0 - c} for c in sorted(set(cids))]
    return ClusterResult(k_optimal=len(set(cids)), silhouette_score=0.5,
                         per_cluster_stats=stats)


@pytest.fixture(autouse=True)
def _receptor(tmp_path):
    (tmp_path / "r.pdb").write_text("END\n")


def test_gating_skips_clashed_poses(tmp_path):
    poses = [_pose(0, -12.0, cid=0), _pose(1, -11.0, clashed=True, cid=1),
             _pose(2, -10.0, cid=2)]
    cfg = _cfg(tmp_path)
    with patch.object(mmgbsa, "compute_mmgbsa_single", return_value=-30.0) as m:
        mmgbsa.refine_topk_poses(poses, _cluster([0, 1, 2]), cfg)
    scored_idx = {c.kwargs.get("pose_pdb").name for c in m.call_args_list}
    # clashed pose 1 must not have been scored
    assert m.call_count == 2
    assert poses[1].mmgbsa_dg is None
    assert poses[0].mmgbsa_dg == -30.0


def test_all_clashed_falls_back_to_scoring_them(tmp_path):
    poses = [_pose(0, -12.0, clashed=True, cid=0), _pose(1, -11.0, clashed=True, cid=1)]
    with patch.object(mmgbsa, "compute_mmgbsa_single", return_value=-20.0) as m:
        mmgbsa.refine_topk_poses(poses, _cluster([0, 1]), _cfg(tmp_path))
    assert m.call_count == 2  # gating left nothing, so both scored


def test_3traj_and_dielectric_threaded_to_compute(tmp_path):
    poses = [_pose(0, -12.0, cid=0)]
    cfg = _cfg(tmp_path, mmgbsa_3traj=True, mmgbsa_solute_dielectric=2.0)
    with patch.object(mmgbsa, "compute_mmgbsa_single", return_value=-25.0) as m:
        mmgbsa.refine_topk_poses(poses, _cluster([0]), cfg)
    kw = m.call_args_list[0].kwargs
    assert kw["three_traj"] is True
    assert kw["solute_dielectric"] == 2.0


def test_ie_adds_signed_entropy_to_dg(tmp_path):
    poses = [_pose(0, -12.0, cid=0)]
    cfg = _cfg(tmp_path, mmgbsa_include_ie=True)
    with patch.object(mmgbsa, "compute_mmgbsa_single", return_value=-30.0), \
         patch("hybridock_pep.scoring.interaction_entropy.sample_interaction_energies",
               return_value=np.array([1.0, -1.0, 2.0, -2.0])), \
         patch("hybridock_pep.scoring.interaction_entropy.interaction_entropy",
               return_value=2.5):
        mmgbsa.refine_topk_poses(poses, _cluster([0]), cfg)
    # ΔG = ΔH(-30) + (−TΔS = +2.5) = -27.5
    assert poses[0].mmgbsa_dg == pytest.approx(-27.5)
