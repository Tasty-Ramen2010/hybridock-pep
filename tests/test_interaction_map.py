"""Unit tests for the typed interaction-fingerprint (crystal-pose IFP) module."""
from __future__ import annotations

import numpy as np
import pytest

from hybridock_pep.scoring.interaction_map import (
    IFP_FEATURE_ORDER,
    clash_metrics,
    compute_ifp,
    interaction_fingerprint,
    ifp_vector,
    receptor_atoms,
    score_crystal_complex,
)


def _write_pdb(path, coords, resname="ALA", chain="A", element="C") -> None:
    """Write heavy atoms at ``coords`` (list of xyz) to a minimal PDB."""
    lines = []
    for i, (x, y, z) in enumerate(coords, start=1):
        lines.append(
            f"ATOM  {i:>5}  CA  {resname} {chain}{i:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}"
        )
    path.write_text("\n".join(lines) + "\n")


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


def test_ranking_confidence_high_when_spread() -> None:
    """A well-spread panel of rank_scores is flagged high-confidence."""
    from hybridock_pep.scoring.interaction_map import ranking_confidence, RANK_CONFIDENCE_SPREAD_THRESHOLD
    flag, spread = ranking_confidence([-9.0, -8.0, -7.0, -6.0])
    assert flag == "high"
    assert spread >= RANK_CONFIDENCE_SPREAD_THRESHOLD


def test_ranking_confidence_low_when_clustered() -> None:
    """Near-identical rank_scores (model can't discriminate) are flagged low-confidence."""
    from hybridock_pep.scoring.interaction_map import ranking_confidence
    flag, spread = ranking_confidence([-8.50, -8.49, -8.51, -8.50])
    assert flag == "low"
    assert spread < 0.40


def test_ranking_confidence_needs_two_scores() -> None:
    """Fewer than two finite scores cannot be ranked -> low confidence, zero spread."""
    from hybridock_pep.scoring.interaction_map import ranking_confidence
    assert ranking_confidence([-8.0]) == ("low", 0.0)
    assert ranking_confidence([]) == ("low", 0.0)


def test_composition_ifp_sums_to_one() -> None:
    """The composition-normalized IFP vector sums to 1 when any contacts are present."""
    from hybridock_pep.scoring.interaction_map import _composition_ifp_vector
    rec = [("pos", "chg", np.array([0.0, 0.0, 0.0])), ("hyd", "hyd", np.array([10.0, 0.0, 0.0]))]
    pep = [("neg", np.array([3.0, 0.0, 0.0])), ("hyd", np.array([11.5, 0.0, 0.0]))]
    f = interaction_fingerprint(rec, pep)
    v = _composition_ifp_vector(f)
    assert v.shape == (19,)
    assert abs(v.sum() - 1.0) < 1e-9


def test_composition_ifp_all_zero_stays_zero() -> None:
    """An empty fingerprint (no contacts) normalizes to all-zero, not a divide-by-zero."""
    from hybridock_pep.scoring.interaction_map import _composition_ifp_vector
    f = {k: 0.0 for k in IFP_FEATURE_ORDER}
    v = _composition_ifp_vector(f)
    assert np.all(v == 0.0)


def test_clash_metrics_zero_for_separated(tmp_path) -> None:
    """A peptide 5 Å from the receptor has no clashing atoms."""
    rec = tmp_path / "rec.pdb"
    pep = tmp_path / "pep.pdb"
    _write_pdb(rec, [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)])
    _write_pdb(pep, [(5.0, 0.0, 0.0), (6.5, 0.0, 0.0)], chain="B")
    n_clash, n_pep, frac = clash_metrics(rec, pep)
    assert n_clash == 0 and n_pep == 2 and frac == 0.0


def test_clash_metrics_counts_overlaps(tmp_path) -> None:
    """Peptide atoms sitting on top of receptor atoms (<2 Å) count as clashes."""
    rec = tmp_path / "rec.pdb"
    pep = tmp_path / "pep.pdb"
    _write_pdb(rec, [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0)])
    _write_pdb(pep, [(0.5, 0.0, 0.0), (3.2, 0.0, 0.0)], chain="B")  # both within 2 Å
    n_clash, n_pep, frac = clash_metrics(rec, pep)
    assert n_clash == 2 and frac == 1.0


def test_score_crystal_complex_refuses_clashing_pose(tmp_path) -> None:
    """A physically impossible (overlapping) pose is refused before it can be scored."""
    rec = tmp_path / "rec.pdb"
    pep = tmp_path / "pep.pdb"
    _write_pdb(rec, [(float(i), 0.0, 0.0) for i in range(10)])
    _write_pdb(pep, [(float(i) + 0.3, 0.0, 0.0) for i in range(10)], chain="B")  # fully overlapping
    with pytest.raises(ValueError, match="clashing"):
        score_crystal_complex(str(rec), str(pep), "AAAAAAAAAA")


def test_allow_clashes_bypasses_guard(tmp_path) -> None:
    """allow_clashes=True skips the guard (returns a value or None, but does not raise ValueError)."""
    rec = tmp_path / "rec.pdb"
    pep = tmp_path / "pep.pdb"
    _write_pdb(rec, [(float(i), 0.0, 0.0) for i in range(10)])
    _write_pdb(pep, [(float(i) + 0.3, 0.0, 0.0) for i in range(10)], chain="B")
    # Must not raise the clash ValueError; geometry may be unavailable on this toy input -> None.
    score_crystal_complex(str(rec), str(pep), "AAAAAAAAAA", allow_clashes=True)
