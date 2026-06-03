"""Tests for the per-family dispatcher (scoring/per_family.py + entropy.py
schema-v3 path)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hybridock_pep.models import ScoredPose
from hybridock_pep.scoring.entropy import (
    apply_calibration, calibration_mode, load_calibration,
)
from hybridock_pep.scoring.per_family import (
    DispatchResult, MIN_FAMILY_SIM, build_family_kmer_index,
    dispatch_per_family, receptor_sequence, _confidence_band,
)


def _atom(name: str, resn: str, chain: str, resseq: int) -> str:
    return (f"ATOM      1  {name:<4s}{resn:<3s} {chain}{resseq:>4d}    "
            f"   0.000   0.000   0.000  1.00  0.00           {name[0]}\n")


@pytest.fixture()
def per_family_cal(tmp_path: Path) -> dict:
    """A toy schema-v3 calibration with 2 families + fallback."""
    return {
        "schema_version": 3,
        "model_type": "per_family_ridge",
        "families": {
            "A": {
                "w_vina": 0.5, "w_ad4": 0.0, "w_contact": -0.2,
                "w_s_ss_weighted": 0.1, "intercept": -10.0,
                "pdbs": ["1AAA", "2AAA", "3AAA"],
            },
            "B": {
                "w_vina": 0.3, "w_ad4": 0.0, "w_contact": -0.3,
                "w_s_ss_weighted": 0.0, "intercept": -5.0,
                "pdbs": ["1BBB", "2BBB"],
            },
        },
        "fallback": {
            "w_vina": 0.2, "w_ad4": 0.0, "w_contact": -0.1,
            "w_s_ss_weighted": 0.0, "intercept": -8.0,
            "pdbs": [],
        },
    }


@pytest.fixture()
def fake_pdb_dir(tmp_path: Path) -> Path:
    """Mock raw_pdbs dir with the family member PDBs."""
    d = tmp_path / "raw_pdbs"
    d.mkdir()
    # Family A members all have a shared 12-aa sequence so k-mers overlap
    for pdb in ("1AAA", "2AAA", "3AAA"):
        body = []
        for i, resn in enumerate(
            ["ALA","LEU","VAL","ILE","PHE","LYS","ARG","ASP","GLU","HIS","TRP","TYR"]):
            body.append(_atom("CA", resn, "A", i+1))
        (d / f"{pdb}.pdb").write_text("".join(body))
    # Family B members share a different sequence
    for pdb in ("1BBB", "2BBB"):
        body = []
        for i, resn in enumerate(
            ["GLY","SER","THR","ASN","GLN","CYS","MET","PRO","TRP","TYR","HIS","ARG"]):
            body.append(_atom("CA", resn, "A", i+1))
        (d / f"{pdb}.pdb").write_text("".join(body))
    return d


class TestCalibrationMode:
    def test_detects_per_family_by_model_type(self) -> None:
        cal = {"model_type": "per_family_ridge", "families": {}, "fallback": {}}
        assert calibration_mode(cal) == "per_family"

    def test_detects_per_family_by_schema_version(self) -> None:
        cal = {"schema_version": 3, "families": {"A": {}}, "fallback": {}}
        assert calibration_mode(cal) == "per_family"

    def test_v2_still_ridge(self) -> None:
        cal = {"schema_version": 2, "w_vina": 0.5, "model_type": "ridge"}
        assert calibration_mode(cal) == "ridge"


class TestReceptorSequence:
    def test_returns_longest_chain(self, tmp_path: Path) -> None:
        pdb = tmp_path / "x.pdb"
        pdb.write_text(
            _atom("CA", "ALA", "A", 1) + _atom("CA", "LEU", "A", 2)
            + _atom("CA", "VAL", "B", 1)
        )
        assert receptor_sequence(pdb) == "AL"  # chain A is length 2

    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        assert receptor_sequence(tmp_path / "nope.pdb") is None


class TestConfidenceBand:
    def test_in_distribution(self) -> None:
        assert _confidence_band(0.25) == "in_distribution"
        assert _confidence_band(0.20) == "in_distribution"

    def test_borderline(self) -> None:
        assert _confidence_band(0.15) == "borderline"
        assert _confidence_band(0.10) == "borderline"

    def test_out_of_distribution(self) -> None:
        assert _confidence_band(0.09) == "out_of_distribution"
        assert _confidence_band(0.0) == "out_of_distribution"


class TestBuildIndex:
    def test_indexes_all_families(self, per_family_cal: dict, fake_pdb_dir: Path) -> None:
        idx = build_family_kmer_index(per_family_cal, fake_pdb_dir)
        assert set(idx.keys()) == {"A", "B"}
        # Each family has the expected number of members with k-mer sets
        assert len(idx["A"]) == 3
        assert len(idx["B"]) == 2
        assert all(len(s) > 0 for s in idx["A"])

    def test_missing_pdbs_silently_skipped(self, per_family_cal: dict,
                                           tmp_path: Path) -> None:
        # Empty dir → no member files found
        empty = tmp_path / "empty"
        empty.mkdir()
        idx = build_family_kmer_index(per_family_cal, empty)
        assert idx == {"A": [], "B": []}


class TestDispatch:
    def test_routes_to_matching_family(self, per_family_cal: dict,
                                       fake_pdb_dir: Path) -> None:
        idx = build_family_kmer_index(per_family_cal, fake_pdb_dir)
        # A query sequence very similar to family A members
        seq = "ALVIFKRDEHWY"
        result = dispatch_per_family(seq, per_family_cal, idx)
        assert isinstance(result, DispatchResult)
        assert result.family_id == "A"
        assert result.similarity >= MIN_FAMILY_SIM
        assert result.confidence_band in ("in_distribution", "borderline")
        # The ridge is family A's
        assert result.ridge["intercept"] == -10.0

    def test_falls_back_when_below_gate(self, per_family_cal: dict,
                                        fake_pdb_dir: Path) -> None:
        idx = build_family_kmer_index(per_family_cal, fake_pdb_dir)
        # Sequence with no overlap with either family
        seq = "XXXXXXXXXXXX"
        result = dispatch_per_family(seq, per_family_cal, idx)
        assert result.family_id == "fallback"
        assert result.confidence_band == "out_of_distribution"
        assert result.ridge["intercept"] == -8.0  # fallback's intercept

    def test_rejects_bad_calibration(self) -> None:
        with pytest.raises(ValueError, match="families.*fallback"):
            dispatch_per_family("AAA", {"w_vina": 1.0}, family_member_kmers={})

    def test_requires_kmer_index(self, per_family_cal: dict) -> None:
        with pytest.raises(ValueError, match="pre-computed"):
            dispatch_per_family("AAA", per_family_cal, family_member_kmers=None)


class TestLoadCalibration:
    def test_loads_v3_per_family_json(self, per_family_cal: dict, tmp_path: Path) -> None:
        # _validate_ridge requires w_ad4 — add it on every ridge
        path = tmp_path / "cal.json"
        path.write_text(json.dumps(per_family_cal))
        loaded = load_calibration(path)
        assert calibration_mode(loaded) == "per_family"
        assert set(loaded["families"].keys()) == {"A", "B"}

    def test_rejects_v3_missing_fallback(self, tmp_path: Path) -> None:
        bad = {
            "schema_version": 3, "model_type": "per_family_ridge",
            "families": {"A": {"w_vina": 0.5, "w_ad4": 0.0, "w_contact": -0.2,
                               "intercept": -10.0}},
            # no fallback
        }
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="fallback"):
            load_calibration(path)


@pytest.fixture()
def valid_receptor(tmp_path: Path) -> Path:
    p = tmp_path / "receptor.pdb"
    p.write_text(_atom("CA", "ALA", "A", 1))
    return p


class TestApplyPerFamily:
    def test_uses_override_when_provided(self, per_family_cal: dict,
                                         tmp_path: Path) -> None:
        # Patch w_ad4 onto each ridge since apply_calibration reads it
        cal = json.loads(json.dumps(per_family_cal))
        for fit in list(cal["families"].values()) + [cal["fallback"]]:
            fit.setdefault("w_ad4", 0.0)
        pose = ScoredPose(
            pose_idx=0, pdb_path=tmp_path / "p.pdb",
            sequence="AA", ca_coords=np.zeros((2, 3)),
            vina_score=-8.0, n_contact_residues=5,
            s_ss_weighted=2.0, s_sc_sum=0.0, s_bb_sum=0.0,
        )
        apply_calibration(
            pose, cal, n_residues=2, n_contact_residues=5,
            ridge_override=cal["families"]["A"],
        )
        # Family A: 0.5*-8 + 0*0 + -0.2*5 + 0.1*2 + -10 = -4 + 0 + -1 + 0.2 + -10 = -14.8
        assert pose.hybrid_score == pytest.approx(-14.8, abs=1e-6)

    def test_uses_fallback_when_no_override(self, per_family_cal: dict,
                                            tmp_path: Path) -> None:
        cal = json.loads(json.dumps(per_family_cal))
        for fit in list(cal["families"].values()) + [cal["fallback"]]:
            fit.setdefault("w_ad4", 0.0)
        pose = ScoredPose(
            pose_idx=0, pdb_path=tmp_path / "p.pdb",
            sequence="AA", ca_coords=np.zeros((2, 3)),
            vina_score=-8.0, n_contact_residues=5,
        )
        apply_calibration(pose, cal, n_residues=2, n_contact_residues=5,
                          ridge_override=None)
        # Fallback: 0.2 * -8 + 0 + -0.1 * 5 + 0 + -8 = -1.6 + -0.5 + -8 = -10.1
        assert pose.hybrid_score == pytest.approx(-10.1, abs=1e-6)

    def test_raises_when_per_family_has_no_fallback(self, tmp_path: Path) -> None:
        bad_cal = {
            "schema_version": 3, "model_type": "per_family_ridge",
            "families": {"A": {"w_vina": 0.5}},
            # no fallback
        }
        pose = ScoredPose(
            pose_idx=0, pdb_path=tmp_path / "p.pdb",
            sequence="AA", ca_coords=np.zeros((2, 3)),
            vina_score=-8.0,
        )
        with pytest.raises(ValueError, match="fallback"):
            apply_calibration(pose, bad_cal, n_residues=2, ridge_override=None)
