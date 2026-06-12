"""Tests for the charge-aware protonation entry and the MD trajectory recorder."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from hybridock_pep.scoring.md_recorder import (dataset_summary, load_dataset,
                                               record_trajectory)
from hybridock_pep.scoring.protonation import (pdb2pqr_available,
                                               titratable_state_summary)


def test_md_recorder_roundtrip(tmp_path: Path) -> None:
    ds = tmp_path / "md.jsonl"
    e = np.array([-50.0, -48.0, -52.0, -49.0, -51.0])
    record_trajectory("1ABC_pose01", "ACDEF", e, minus_tds=1.23,
                      ca_rmsf=[0.5, 0.8, 0.6], static_features={"hyd_frac": 0.4},
                      dataset_path=ds)
    record_trajectory("1ABC_pose02", "ACDEF", e, minus_tds=2.34, dataset_path=ds)
    recs = load_dataset(ds)
    assert len(recs) == 2
    assert recs[0]["n_frames"] == 5
    assert recs[0]["e_int_series"][0] == -50.0
    assert recs[0]["minus_tds"] == 1.23
    summ = dataset_summary(ds)
    assert summ["n_records"] == 2
    assert summ["unique_complexes"] == 1  # both poses of 1ABC


def test_md_recorder_creates_dir(tmp_path: Path) -> None:
    ds = tmp_path / "sub" / "md.jsonl"
    record_trajectory("x", "AA", [1.0, 2.0], 0.5, dataset_path=ds)
    assert ds.exists()


def test_titratable_summary_counts_residues(tmp_path: Path) -> None:
    pqr = tmp_path / "t.pqr"
    pqr.write_text(
        "ATOM      1  N   ASP A   1       0.0   0.0   0.0  0.1 1.8\n"
        "ATOM      2  CA  ASP A   1       1.0   0.0   0.0  0.1 1.9\n"
        "ATOM      3  N   HIP A   2       2.0   0.0   0.0  0.1 1.8\n"
        "ATOM      4  N   ALA A   3       3.0   0.0   0.0  0.1 1.8\n"
    )
    s = titratable_state_summary(pqr)
    assert s.get("ASP") == 1
    assert s.get("HIP") == 1  # protonated-His variant counted
    assert "ALA" not in s     # non-titratable excluded


def test_pdb2pqr_available_is_bool() -> None:
    assert isinstance(pdb2pqr_available(), bool)
