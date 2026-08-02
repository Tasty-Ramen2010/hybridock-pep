from __future__ import annotations

import subprocess
from pathlib import Path

from hybridock_pep.toolpath import which as _which


def convert_pdb_to_pdbqt(
    pdb_path: Path, pdbqt_path: Path, *, add_hydrogens: bool = True
) -> subprocess.CompletedProcess:
    """Convert a PDB to a flat (no ROOT/ENDROOT) rigid PDBQT.

    Prefers ADFRsuite's `babel` (Open Babel 2.x, bundled with ADFRsuite).
    Falls back to conda-forge's `obabel` (Open Babel 3.x, a score-env
    dependency) when ADFRsuite is absent — e.g. on Apple Silicon, where
    ADFRsuite ships only an x86_64 tarball and needs Rosetta 2. The two
    binaries take incompatible CLI syntax, and obabel does not default to
    Gasteiger charges the way babel does, so callers should go through this
    function rather than invoking either binary directly.

    Args:
        pdb_path: Source PDB file.
        pdbqt_path: Destination PDBQT path.
        add_hydrogens: Whether to pass the hydrogen-addition flag.

    Returns:
        The completed subprocess result. Callers check returncode and
        pdbqt_path's existence/size themselves (matches existing call sites).

    Raises:
        FileNotFoundError: If neither babel nor obabel is on PATH.
    """
    babel_bin = _which("babel")
    if babel_bin is not None:
        cmd = [babel_bin, "-i", "pdb", str(pdb_path), "-o", "pdbqt", str(pdbqt_path)]
        if add_hydrogens:
            cmd.append("-h")
        cmd.append("-xr")
        return subprocess.run(cmd, capture_output=True, text=True)

    obabel_bin = _which("obabel")
    if obabel_bin is not None:
        cmd = [obabel_bin, "-ipdb", str(pdb_path), "-opdbqt", "-O", str(pdbqt_path)]
        if add_hydrogens:
            cmd.append("-h")
        cmd += ["-xr", "--partialcharge", "gasteiger"]
        return subprocess.run(cmd, capture_output=True, text=True)

    raise FileNotFoundError(
        "Neither babel (ADFRsuite) nor obabel (conda-forge openbabel) found on "
        "PATH. Install ADFRsuite, or add openbabel to score-env.yml."
    )
