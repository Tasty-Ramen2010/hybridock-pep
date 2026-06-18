"""Unit tests for the typed interaction-fingerprint (crystal-pose IFP) module."""
from __future__ import annotations

import numpy as np

from hybridock_pep.scoring.interaction_map import (
    IFP_FEATURE_ORDER,
    compute_ifp,
    interaction_fingerprint,
    ifp_vector,
    receptor_atoms,
)


def test_feature_order_is_19() -> None:
    assert len(IFP_FEATURE_ORDER) == 19
    assert "sb_fav" in IFP_FEATURE_ORDER and "hb_to_chg" in IFP_FEATURE_ORDER


def test_favorable_salt_bridge_detected() -> None:
    """A receptor cation 3.0 Å from a peptide anion registers one favorable salt bridge."""
    rec = [("pos", "chg", np.array([0.0, 0.0, 0.0]))]   # Lys NZ
    pep = [("neg", np.array([3.0, 0.0, 0.0]))]           # Asp/Glu carboxylate
    f = interaction_fingerprint(rec, pep)
    assert f["sb_fav"] == 1.0
    assert f["sb_fav_str"] > 0.0
    assert f["sb_unfav"] == 0.0
    assert f["contact_chg"] > 0.0


def test_like_charge_repulsion_is_unfavorable() -> None:
    rec = [("pos", "chg", np.array([0.0, 0.0, 0.0]))]
    pep = [("pos", np.array([3.0, 0.0, 0.0]))]
    f = interaction_fingerprint(rec, pep)
    assert f["sb_unfav"] == 1.0
    assert f["sb_fav"] == 0.0


def test_hbond_typed_by_receptor_residue() -> None:
    """A peptide donor 3.0 Å from a polar receptor acceptor is an H-bond typed 'pol'."""
    rec = [("acc", "pol", np.array([0.0, 0.0, 0.0]))]    # e.g. Ser OG
    pep = [("don", np.array([3.0, 0.0, 0.0]))]           # backbone N
    f = interaction_fingerprint(rec, pep)
    assert f["hbond"] == 1.0
    assert f["hb_to_pol"] == 1.0
    assert f["hb_to_chg"] == 0.0


def test_distance_cutoff_excludes_far_contacts() -> None:
    rec = [("pos", "chg", np.array([0.0, 0.0, 0.0]))]
    pep = [("neg", np.array([9.0, 0.0, 0.0]))]           # > 6 Å
    f = interaction_fingerprint(rec, pep)
    assert all(v == 0.0 for v in f.values())


def test_ifp_vector_matches_order() -> None:
    rec = [("pos", "chg", np.array([0.0, 0.0, 0.0]))]
    pep = [("neg", np.array([3.0, 0.0, 0.0]))]
    f = interaction_fingerprint(rec, pep)
    v = ifp_vector(f)
    assert v.shape == (19,)
    assert v[IFP_FEATURE_ORDER.index("sb_fav")] == 1.0


def test_compute_ifp_roundtrip_from_pdb(tmp_path) -> None:
    """End-to-end parse: write a tiny receptor + peptide PDB and recover the salt bridge."""
    rec_pdb = tmp_path / "rec.pdb"
    pep_pdb = tmp_path / "pep.pdb"
    # Lys NZ at origin
    rec_pdb.write_text(
        "ATOM      1  NZ  LYS A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    )
    # Asp carboxylate O 3 Å away (peptide)
    pep_pdb.write_text(
        "ATOM      1  OD1 ASP B   1       3.000   0.000   0.000  1.00  0.00           O\n"
    )
    assert len(receptor_atoms(rec_pdb)) == 1
    f = compute_ifp(rec_pdb, pep_pdb)
    assert f["sb_fav"] == 1.0
