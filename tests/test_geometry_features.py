"""Unit tests for geometry+MJ feature extraction and the MJ contact potential."""
from __future__ import annotations

from pathlib import Path

import pytest

from hybridock_pep.scoring.geometry_features import (
    GEOMETRY_FEATURE_KEYS,
    compute_geometry_features,
)
from hybridock_pep.scoring.mj_potential import MJ_ENERGY


def test_mj_potential_is_complete_and_symmetric() -> None:
    aas = "ACDEFGHIKLMNPQRSTVWY"
    for a in aas:
        for b in aas:
            assert (a, b) in MJ_ENERGY, f"missing MJ pair {a}{b}"
            assert MJ_ENERGY[(a, b)] == pytest.approx(MJ_ENERGY[(b, a)])


def test_mj_hydrophobic_more_favorable_than_charged() -> None:
    # Trp-Trp hydrophobic burial should be far more favourable (more negative)
    # than Lys-Lys like-charge contact — the hotspot signal the feature encodes.
    assert MJ_ENERGY[("W", "W")] < MJ_ENERGY[("K", "K")]
    assert MJ_ENERGY[("F", "L")] < MJ_ENERGY[("D", "K")] or MJ_ENERGY[("F", "L")] < 0


def _make_complex(tmp_path: Path) -> tuple[Path, Path]:
    """Write a minimal peptide pose + receptor PDB with a couple of close residues."""
    pep = tmp_path / "pep.pdb"
    rec = tmp_path / "rec.pdb"
    # peptide: two residues (TRP, LEU) along x; receptor: two residues nearby
    pep.write_text(
        "ATOM      1  CA  TRP A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CB  TRP A   1       1.500   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  CA  LEU A   2       3.800   0.000   0.000  1.00  0.00           C\n"
        "ATOM      4  CB  LEU A   2       5.300   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    rec.write_text(
        "ATOM      1  CA  PHE B   1       2.000   3.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CB  PHE B   1       2.000   4.500   0.000  1.00  0.00           C\n"
        "ATOM      3  CA  ASP B   2       4.000   3.000   0.000  1.00  0.00           C\n"
        "ATOM      4  CB  ASP B   2       4.000   4.500   0.000  1.00  0.00           C\n"
        "END\n"
    )
    return pep, rec


def test_compute_geometry_features_returns_all_keys(tmp_path: Path) -> None:
    pep, rec = _make_complex(tmp_path)
    feats = compute_geometry_features(pep, rec)
    assert feats is not None
    for k in GEOMETRY_FEATURE_KEYS:
        assert k in feats
        assert isinstance(feats[k], float)


def test_mj_contact_is_negative_for_real_contacts(tmp_path: Path) -> None:
    # peptide and receptor residues are within contact distance -> MJ sum should be
    # populated and favourable (negative).
    pep, rec = _make_complex(tmp_path)
    feats = compute_geometry_features(pep, rec)
    assert feats is not None
    assert feats["mj_contact"] < 0.0


def test_rg_per_l_extended_exceeds_compact(tmp_path: Path) -> None:
    """rg_per_L must be larger for an extended peptide than a compact one (extendedness proxy)."""
    from hybridock_pep.scoring.geometry_features import _rg_per_residue

    extended = tmp_path / "ext.pdb"
    compact = tmp_path / "cmp.pdb"
    # extended: 4 CA strung along x at 3.8 Å spacing
    extended.write_text("".join(
        f"ATOM  {i:5d}  CA  ALA A{i:4d}    {i*3.8:8.3f}   0.000   0.000  1.00  0.00           C\n"
        for i in range(1, 5)) + "END\n")
    # compact: 4 CA clustered in a tight tetrahedron-ish cloud
    compact.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CA  ALA A   2       1.500   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  CA  ALA A   3       0.750   1.300   0.000  1.00  0.00           C\n"
        "ATOM      4  CA  ALA A   4       0.750   0.430   1.220  1.00  0.00           C\n"
        "END\n")
    assert _rg_per_residue(extended) > _rg_per_residue(compact) > 0.0


def test_conformational_entropy_penalty_positive_and_scales(tmp_path: Path) -> None:
    """The MM-GBSA conformational penalty must be >=0 and scale with alpha (weaker for extended)."""
    from hybridock_pep.scoring.mmgbsa import conformational_entropy_penalty

    pep = tmp_path / "pep.pdb"
    pep.write_text("".join(
        f"ATOM  {i:5d}  CA  ALA A{i:4d}    {i*3.8:8.3f}   0.000   0.000  1.00  0.00           C\n"
        for i in range(1, 6)) + "END\n")
    p1 = conformational_entropy_penalty(pep, alpha=5.4)
    p2 = conformational_entropy_penalty(pep, alpha=10.8)
    assert p1 > 0.0
    assert p2 == pytest.approx(2.0 * p1, rel=1e-6)


def test_no_interface_returns_none(tmp_path: Path) -> None:
    # receptor placed 100 Å away -> no contacts, no buried SASA on the receptor side,
    # but pocket descriptors require receptor residues near the peptide; expect None.
    pep, rec = _make_complex(tmp_path)
    far = tmp_path / "far.pdb"
    far.write_text(
        "ATOM      1  CA  PHE B   1     200.000 200.000 200.000  1.00  0.00           C\n"
        "END\n"
    )
    feats = compute_geometry_features(pep, far)
    assert feats is None
