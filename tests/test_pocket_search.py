"""_crop_receptor_to_site() -- pocket-search Stage 1's receptor-cropping
helper (sampling/pocket_search.py).

Regression coverage for the TER-record bug: whole-residue cropping drops
residues from the middle of a chain (anything far enough from the pocket
centroid, regardless of neighbors), leaving a gap in otherwise-contiguous
residue numbering. The cropper used to strip every TER record unconditionally
and never write a replacement, so a cropped receptor with such a gap had
nothing in the file explaining the discontinuity -- MDAnalysis's PDB reader
(used deep inside RAPiDock) mis-parses that and fails with an opaque
IndexError reading the occupancy column. Caught live by the blind-docking
flag-combo matrix sweep as "All N candidate pocket refinements failed".
"""
from __future__ import annotations

import numpy as np

from hybridock_pep.sampling.pocket_search import _crop_receptor_to_site


def _atom_line(serial: int, resname: str, chain: str, resseq: int, x: float, y: float, z: float) -> str:
    return (
        f"ATOM  {serial:5d}  CA  {resname:<3s} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
    )


def _write_pdb(path, lines):
    path.write_text("".join(lines))
    return path


class TestCropReceptorToSite:
    def test_keeps_only_residues_within_radius(self, tmp_path):
        lines = [
            _atom_line(1, "ALA", "A", 1, 0.0, 0.0, 0.0),   # inside
            _atom_line(2, "GLY", "A", 2, 0.0, 0.0, 0.0),   # inside
            _atom_line(3, "SER", "A", 50, 100.0, 100.0, 100.0),  # far -- dropped
        ]
        src = _write_pdb(tmp_path / "full.pdb", lines)
        dest = _crop_receptor_to_site(src, np.array([0.0, 0.0, 0.0]), 5.0, tmp_path / "cropped.pdb")
        text = dest.read_text()
        assert " A   1" in text or "A   1" in text
        assert "SER" not in text

    def test_inserts_ter_at_a_mid_chain_gap(self, tmp_path):
        """A residue dropped from the middle of an otherwise-contiguous chain
        must leave a TER marking the discontinuity -- not a silent jump in
        resSeq, which is exactly what confused MDAnalysis's PDB reader."""
        lines = [
            _atom_line(1, "ALA", "A", 1, 0.0, 0.0, 0.0),    # kept
            _atom_line(2, "GLY", "A", 2, 0.0, 0.0, 0.0),    # kept
            _atom_line(3, "SER", "A", 3, 100.0, 100.0, 100.0),  # dropped (far) -- creates the gap
            _atom_line(4, "VAL", "A", 4, 0.0, 0.0, 0.0),    # kept
        ]
        src = _write_pdb(tmp_path / "full.pdb", lines)
        dest = _crop_receptor_to_site(src, np.array([0.0, 0.0, 0.0]), 5.0, tmp_path / "cropped.pdb")
        text = dest.read_text()
        out_lines = [ln for ln in text.splitlines() if ln.strip()]
        # A TER line must separate residue 2 (GLY) from residue 4 (VAL).
        gly_idx = next(i for i, ln in enumerate(out_lines) if "GLY" in ln)
        val_idx = next(i for i, ln in enumerate(out_lines) if "VAL" in ln)
        assert any(out_lines[i].startswith("TER") for i in range(gly_idx + 1, val_idx))

    def test_no_ter_for_contiguous_kept_residues(self, tmp_path):
        lines = [
            _atom_line(1, "ALA", "A", 1, 0.0, 0.0, 0.0),
            _atom_line(2, "GLY", "A", 2, 0.0, 0.0, 0.0),
            _atom_line(3, "VAL", "A", 3, 0.0, 0.0, 0.0),
        ]
        src = _write_pdb(tmp_path / "full.pdb", lines)
        dest = _crop_receptor_to_site(src, np.array([0.0, 0.0, 0.0]), 5.0, tmp_path / "cropped.pdb")
        out_lines = [ln for ln in dest.read_text().splitlines() if ln.strip()]
        # Only the final end-of-chain TER, nothing between contiguous residues.
        ter_lines = [ln for ln in out_lines if ln.startswith("TER")]
        assert len(ter_lines) == 1
        assert out_lines[-1].startswith("TER")

    def test_ends_with_ter_when_any_residue_kept(self, tmp_path):
        lines = [_atom_line(1, "ALA", "A", 1, 0.0, 0.0, 0.0)]
        src = _write_pdb(tmp_path / "full.pdb", lines)
        dest = _crop_receptor_to_site(src, np.array([0.0, 0.0, 0.0]), 5.0, tmp_path / "cropped.pdb")
        out_lines = [ln for ln in dest.read_text().splitlines() if ln.strip()]
        assert out_lines[-1].startswith("TER")

    def test_no_trailing_ter_when_nothing_kept(self, tmp_path):
        lines = [_atom_line(1, "ALA", "A", 1, 100.0, 100.0, 100.0)]  # far
        src = _write_pdb(tmp_path / "full.pdb", lines)
        dest = _crop_receptor_to_site(src, np.array([0.0, 0.0, 0.0]), 5.0, tmp_path / "cropped.pdb")
        assert dest.read_text().strip() == ""

    def test_ter_inserted_between_chains(self, tmp_path):
        lines = [
            _atom_line(1, "ALA", "A", 1, 0.0, 0.0, 0.0),
            _atom_line(2, "GLY", "B", 1, 0.0, 0.0, 0.0),
        ]
        src = _write_pdb(tmp_path / "full.pdb", lines)
        dest = _crop_receptor_to_site(src, np.array([0.0, 0.0, 0.0]), 5.0, tmp_path / "cropped.pdb")
        out_lines = [ln for ln in dest.read_text().splitlines() if ln.strip()]
        ala_idx = next(i for i, ln in enumerate(out_lines) if "ALA" in ln)
        gly_idx = next(i for i, ln in enumerate(out_lines) if "GLY" in ln)
        assert any(out_lines[i].startswith("TER") for i in range(ala_idx + 1, gly_idx))

    def test_original_ter_records_are_not_duplicated(self, tmp_path):
        """The source file's own TER lines are dropped and replaced with
        freshly computed ones -- must not end up with both."""
        lines = [
            _atom_line(1, "ALA", "A", 1, 0.0, 0.0, 0.0),
            "TER\n",
            _atom_line(2, "GLY", "A", 2, 0.0, 0.0, 0.0),
        ]
        src = _write_pdb(tmp_path / "full.pdb", lines)
        dest = _crop_receptor_to_site(src, np.array([0.0, 0.0, 0.0]), 5.0, tmp_path / "cropped.pdb")
        ter_count = sum(1 for ln in dest.read_text().splitlines() if ln.startswith("TER"))
        # Contiguous resSeq 1->2 in the same chain: no mid-chain break, just
        # the one trailing end-of-chain TER.
        assert ter_count == 1
