"""The declash step: a peptide that sits inside the protein is slid out; one that is clear is left alone."""
from pathlib import Path

import numpy as np

from hybridock_pep.scoring.declash import CLEAR, declash_pose, declash_poses_batch


def _pdb(path: Path, coords: np.ndarray) -> Path:
    lines = [
        f"ATOM  {i + 1:5d}  CA  ALA A{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        for i, (x, y, z) in enumerate(coords)
    ]
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


def _slab() -> np.ndarray:
    """A flat 'protein' wall of atoms in the z=0 plane, 2 Å apart."""
    xs = np.arange(-20, 21, 2.0)
    return np.array([(x, y, 0.0) for x in xs for y in xs])


def _peptide(z: float) -> np.ndarray:
    return np.array([(x, 0.0, z) for x in np.arange(-6, 6.1, 3.0)])


def test_overlapping_pose_is_moved_clear_without_changing_its_shape(tmp_path):
    rec = _slab()
    src = _pdb(tmp_path / "in.pdb", _peptide(0.4))
    info = declash_pose(src, rec, tmp_path / "out.pdb")
    assert info["clear"] and info["clearance"] >= CLEAR and info["moved"] > 1.0
    before = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in src.read_text().splitlines() if l.startswith("ATOM")])
    after = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in (tmp_path / "out.pdb").read_text().splitlines() if l.startswith("ATOM")])
    d0 = np.linalg.norm(before[:, None] - before[None], axis=2)
    d1 = np.linalg.norm(after[:, None] - after[None], axis=2)
    assert np.allclose(d0, d1, atol=0.02), "the peptide must move as one rigid body"


def test_a_clear_pose_is_left_exactly_as_it_was(tmp_path):
    rec = _slab()
    src = _pdb(tmp_path / "in.pdb", _peptide(5.0))
    info = declash_pose(src, rec, tmp_path / "out.pdb")
    assert info["moved"] == 0.0 and info["clear"]
    assert (tmp_path / "out.pdb").read_text() == src.read_text().rstrip("\n") + "\n"


def test_batch_never_loses_a_pose(tmp_path):
    rec = _slab()
    good = _pdb(tmp_path / "pose_0.pdb", _peptide(0.2))
    bad = tmp_path / "pose_1.pdb"
    bad.write_text("not a pdb\n")
    out = declash_poses_batch([good, bad], rec, tmp_path / "clear")
    assert [p.name for p in out] == ["pose_0.pdb", "pose_1.pdb"] and all(p.exists() for p in out)
