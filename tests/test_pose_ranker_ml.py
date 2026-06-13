"""Tests for the ML pose ranker (structural pose ranking; never touches affinity)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hybridock_pep.models import ScoredPose
from hybridock_pep.output.csv_writer import _rank_key
from hybridock_pep.scoring import pose_ranker_ml as prm


def _pose(idx: int, pdb: Path, **kw) -> ScoredPose:
    return ScoredPose(
        pose_idx=idx, pdb_path=pdb, sequence="AAA",
        ca_coords=np.zeros((3, 3), dtype=float), **kw,
    )


def test_feature_names_match_vector_length():
    assert len(prm.FEATURE_NAMES) == len(prm.RAMA_NAMES) + len(prm.SHAPE_NAMES) == 19


def test_rank_key_prefers_ml_over_bsa(tmp_path):
    """ML score, when present, drives ranking; BSA-fit only breaks ties when ML is absent."""
    pdb = tmp_path / "x.pdb"
    pdb.write_text("END\n")
    # ML says pose A is better (lower predicted RMSD) even though its BSA-fit is worse.
    a = _pose(0, pdb, ml_pose_score=1.5, bsa_fit_score=2.0)
    b = _pose(1, pdb, ml_pose_score=3.0, bsa_fit_score=-1.0)
    assert _rank_key(a) < _rank_key(b)


def test_rank_key_falls_back_to_bsa_then_hybrid(tmp_path):
    pdb = tmp_path / "x.pdb"
    pdb.write_text("END\n")
    assert _rank_key(_pose(0, pdb, bsa_fit_score=-2.0)) == -2.0          # ML absent → BSA
    assert _rank_key(_pose(1, pdb, hybrid_score=-5.0)) == -5.0           # ML+BSA absent → hybrid
    assert _rank_key(_pose(2, pdb)) == float("inf")                      # nothing → inf (sorts last)


def test_missing_artifact_is_silent_noop(tmp_path):
    """No artifact → returns False, leaves ml_pose_score None, never raises."""
    pdb = tmp_path / "x.pdb"
    pdb.write_text("END\n")
    poses = [_pose(0, pdb)]
    ok = prm.compute_ml_pose_scores(poses, model_path=tmp_path / "does_not_exist.joblib")
    assert ok is False
    assert poses[0].ml_pose_score is None


def test_corrupt_artifact_is_silent_noop(tmp_path):
    bad = tmp_path / "bad.joblib"
    bad.write_text("not a joblib file")
    pdb = tmp_path / "x.pdb"
    pdb.write_text("END\n")
    poses = [_pose(0, pdb)]
    assert prm.compute_ml_pose_scores(poses, model_path=bad) is False
    assert poses[0].ml_pose_score is None


def test_affinity_fields_untouched_by_ranker(tmp_path):
    """The ranker must never write to any affinity field (the E94 wall)."""
    pdb = tmp_path / "x.pdb"
    pdb.write_text("END\n")
    p = _pose(0, pdb, hybrid_score=-7.0, vina_score=-6.0, ensemble_dg=-7.5, mmgbsa_dg=-8.0)
    prm.compute_ml_pose_scores([p], model_path=tmp_path / "missing.joblib")
    assert (p.hybrid_score, p.vina_score, p.ensemble_dg, p.mmgbsa_dg) == (-7.0, -6.0, -7.5, -8.0)


@pytest.mark.skipif(not prm.DEFAULT_MODEL_PATH.exists(), reason="model artifact not built")
def test_real_artifact_predicts_positive_rmsd(tmp_path):
    """With the shipped artifact, a real peptide PDB gets a plausible predicted RMSD (>0)."""
    import joblib

    bundle = joblib.load(prm.DEFAULT_MODEL_PATH)
    assert set(("phi_kde", "psi_kde", "model", "feature_names")) <= set(bundle)
    # a minimal tri-peptide backbone+CB so RDKit + Biopython both parse it
    pdb = tmp_path / "tri.pdb"
    pdb.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C\n"
        "ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O\n"
        "ATOM      5  N   ALA A   2       3.332   1.540   0.000  1.00  0.00           N\n"
        "ATOM      6  CA  ALA A   2       3.987   2.840   0.000  1.00  0.00           C\n"
        "ATOM      7  C   ALA A   2       5.500   2.680   0.000  1.00  0.00           C\n"
        "ATOM      8  O   ALA A   2       6.000   1.560   0.000  1.00  0.00           O\n"
        "ATOM      9  N   ALA A   3       6.220   3.790   0.000  1.00  0.00           N\n"
        "ATOM     10  CA  ALA A   3       7.680   3.790   0.000  1.00  0.00           C\n"
        "ATOM     11  C   ALA A   3       8.230   5.210   0.000  1.00  0.00           C\n"
        "ATOM     12  O   ALA A   3       7.470   6.180   0.000  1.00  0.00           O\n"
        "END\n"
    )
    feats = prm.compute_features(pdb, bundle["phi_kde"], bundle["psi_kde"])
    if feats is None:
        pytest.skip("toy peptide has no computable φ/ψ pair")
    pred = float(bundle["model"].predict(np.array([feats]))[0])
    assert 0.0 < pred < 25.0
