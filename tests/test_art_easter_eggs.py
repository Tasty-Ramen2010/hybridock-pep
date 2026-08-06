"""The mosquito easter egg (triggered by the parent project's own peptide,
LISDAELEAIFEADC) was removed by request. This pins that it stays gone and
that the one remaining trigger (IGEM) still works.
"""

from __future__ import annotations

from hybridock_pep.output import art


def test_parent_peptide_no_longer_triggers_anything():
    assert art.easter_egg_for_peptide(art.PARENT_PROJECT_PEPTIDE) is None


def test_igem_still_triggers_the_remaining_easter_egg():
    assert art.easter_egg_for_peptide("IGEM") is not None
    assert art.easter_egg_for_peptide("igem") is not None  # case-insensitive


def test_no_mosquito_art_left_in_the_module():
    assert not hasattr(art, "_MOSQUITO")


def test_unrelated_sequences_still_trigger_nothing():
    assert art.easter_egg_for_peptide("ETFSDLWKLLPE") is None
    assert art.easter_egg_for_peptide("") is None
