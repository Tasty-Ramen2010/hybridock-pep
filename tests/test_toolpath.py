"""Tools installed into our own conda env must be findable without activation.

score-env.yml puts meeko, autogrid4 and openbabel in `score-env/bin`, and the
console script the user runs lives there too. Running it by absolute path —
what install.sh, launch_ui.sh, the TUI and `ssh host hybridock-pep ...` all do —
leaves that directory off $PATH, so `shutil.which` reported every one of those
tools missing on an otherwise correct install and prep fell through to a
fallback that was not installed.

Measured on a DGX Spark: mk_prepare_receptor.py present in score-env/bin,
receptor prep reporting "no receptor-prep tool found".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from hybridock_pep.toolpath import which


def _make_exe(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / name
    p.write_text("#!/bin/sh\nexit 0\n")
    p.chmod(0o755)
    return p


class TestWhich:
    def test_finds_a_tool_on_path(self, tmp_path, monkeypatch):
        d = tmp_path / "onpath"
        exe = _make_exe(d, "mk_prepare_receptor.py")
        monkeypatch.setenv("PATH", str(d))

        assert which("mk_prepare_receptor.py") == str(exe)

    def test_finds_a_tool_in_the_running_env_bin_when_path_misses_it(
        self, tmp_path, monkeypatch
    ):
        """The regression: the tool is in sys.prefix/bin and PATH is elsewhere."""
        prefix = tmp_path / "envs" / "score-env"
        exe = _make_exe(prefix / "bin", "mk_prepare_receptor.py")
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setattr(sys, "prefix", str(prefix))

        assert which("mk_prepare_receptor.py") == str(exe)

    def test_path_wins_over_the_env_bin(self, tmp_path, monkeypatch):
        """An explicitly activated env or a system install must still take
        precedence — this only ever adds a fallback."""
        on_path = _make_exe(tmp_path / "onpath", "autogrid4")
        prefix = tmp_path / "envs" / "score-env"
        _make_exe(prefix / "bin", "autogrid4")
        monkeypatch.setenv("PATH", str(tmp_path / "onpath"))
        monkeypatch.setattr(sys, "prefix", str(prefix))

        assert which("autogrid4") == str(on_path)

    def test_returns_none_when_genuinely_absent(self, tmp_path, monkeypatch):
        prefix = tmp_path / "envs" / "score-env"
        (prefix / "bin").mkdir(parents=True)
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setattr(sys, "prefix", str(prefix))

        assert which("definitely-not-a-real-tool") is None

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics")
    def test_ignores_a_non_executable_file(self, tmp_path, monkeypatch):
        prefix = tmp_path / "envs" / "score-env"
        (prefix / "bin").mkdir(parents=True)
        (prefix / "bin" / "obabel").write_text("not executable")
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setattr(sys, "prefix", str(prefix))

        assert which("obabel") is None


class TestPrepModulesUseIt:
    """Every prep module must route through the env-aware lookup.

    A single `shutil.which` left behind reintroduces the bug for that one tool,
    silently, on exactly the installs that are hardest to debug.
    """

    MODULES = [
        "src/hybridock_pep/prep/receptor.py",
        "src/hybridock_pep/prep/ligand.py",
        "src/hybridock_pep/prep/grids.py",
        "src/hybridock_pep/prep/pdbqt_convert.py",
        "src/hybridock_pep/scoring/protonation.py",
    ]

    @pytest.mark.parametrize("rel", MODULES)
    def test_no_bare_shutil_which(self, rel):
        repo = Path(__file__).resolve().parent.parent
        text = (repo / rel).read_text()
        assert "shutil.which(" not in text, (
            f"{rel} calls shutil.which directly — it will not see tools "
            "installed in score-env/bin. Use hybridock_pep.toolpath.which."
        )


class TestMeekoLigandFallback:
    """ARM Linux has no conda-forge openbabel for py3.11, so babel/obabel are
    both absent and every pose failed ligand prep. Meeko's Polymer route works
    there — but must stay a fallback, since the shipped calibration was fitted
    against babel output."""

    def test_babel_available_reports_both_binaries(self, monkeypatch):
        import hybridock_pep.prep.pdbqt_convert as convert_mod

        monkeypatch.setattr(convert_mod, "_which", lambda n: None)
        assert convert_mod.babel_available() is False

        monkeypatch.setattr(
            convert_mod, "_which", lambda n: "/x/obabel" if n == "obabel" else None
        )
        assert convert_mod.babel_available() is True

        monkeypatch.setattr(
            convert_mod, "_which", lambda n: "/x/babel" if n == "babel" else None
        )
        assert convert_mod.babel_available() is True

    def test_meeko_used_when_no_converter_binary_exists(self, tmp_path, monkeypatch):
        import hybridock_pep.prep.ligand as ligand_mod

        monkeypatch.setattr(ligand_mod, "babel_available", lambda: False)
        called = []
        monkeypatch.setattr(
            "hybridock_pep.prep.phospho.prepare_phospho_ligand",
            lambda idx, p, out, stage="prep_phospho": called.append(idx) or (out / "x.pdbqt"),
        )
        pdb = tmp_path / "pose_0.pdb"
        pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\nEND\n")

        ligand_mod._prepare_single_ligand((0, pdb, tmp_path))

        assert called == [0], "no converter binary present but Meeko was not used"

    def test_babel_still_wins_when_present(self, tmp_path, monkeypatch):
        """Machines with babel must keep producing byte-identical PDBQT."""
        import hybridock_pep.prep.ligand as ligand_mod

        monkeypatch.setattr(ligand_mod, "babel_available", lambda: True)
        called = []
        monkeypatch.setattr(
            "hybridock_pep.prep.phospho.prepare_phospho_ligand",
            lambda idx, p, out, stage="prep_phospho": called.append(idx),
        )
        converted = []

        def fake_convert(src, dst, add_hydrogens=True):
            converted.append(dst)
            dst.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n")
            import subprocess
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        monkeypatch.setattr(ligand_mod, "convert_pdb_to_pdbqt", fake_convert)
        pdb = tmp_path / "pose_0.pdb"
        pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\nEND\n")

        ligand_mod._prepare_single_ligand((0, pdb, tmp_path))

        assert not called, "Meeko was used even though babel is available"
        assert converted, "babel path was not taken"

    def test_fallback_failure_reports_the_ordinary_prep_stage(self, tmp_path, monkeypatch):
        """A failure on the Meeko fallback is an ordinary ligand-prep failure.

        Reporting stage="prep_phospho" for a peptide with no phospho residues
        sends whoever reads the failure record chasing the wrong thing.
        """
        import hybridock_pep.prep.ligand as ligand_mod
        from hybridock_pep.models import PoseFailure

        monkeypatch.setattr(ligand_mod, "babel_available", lambda: False)
        result = ligand_mod._prepare_single_ligand((7, tmp_path / "missing.pdb", tmp_path))

        assert isinstance(result, PoseFailure)
        assert result.pose_idx == 7
        assert result.stage == "prep"


class TestToolsAreInvokedByResolvedPath:
    """Locating a tool is not enough — the command must use the path found.

    receptor.py located mk_prepare_receptor.py through the env-aware lookup and
    then ran subprocess with the bare name, so execve searched $PATH again and
    raised FileNotFoundError on a binary the code had just confirmed exists.
    """

    def test_meeko_command_uses_the_resolved_binary(self, tmp_path, monkeypatch):
        import subprocess as sp

        import hybridock_pep.prep.receptor as receptor_mod

        seen = []

        def fake_run(cmd, **kw):
            seen.append(cmd)
            Path(cmd[cmd.index("-o") + 1] + ".pdbqt").write_text("ATOM\n")
            return sp.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(receptor_mod.subprocess, "run", fake_run)
        out = tmp_path / "receptor.pdbqt"
        receptor_mod._try_meeko(tmp_path / "in.pdb", out, "/opt/env/bin/mk_prepare_receptor.py")

        assert seen[0][0] == "/opt/env/bin/mk_prepare_receptor.py", (
            f"invoked by bare name: {seen[0][0]!r} — execve will search $PATH "
            "and miss score-env/bin"
        )

    def test_autogrid_command_uses_the_resolved_binary(self, monkeypatch):
        import hybridock_pep.prep.grids as grids_mod

        src = Path(grids_mod.__file__).read_text()
        assert '["autogrid4", "-p"' not in src, (
            "autogrid4 invoked by bare name — resolve it through toolpath.which"
        )


class TestSidecarEnv:
    """Some tools cannot live in score-env at all and get their own env.

    openbabel is the case: conda-forge's only linux-aarch64 builds are against
    Python 3.9, so it will never enter a 3.11 score-env, and without it ARM
    Linux has no PDBQT converter. It installs fine in an env of its own, and
    `obabel` is a binary — what Python it was built against is irrelevant.
    """

    def test_finds_a_tool_in_the_sidecar_env(self, tmp_path, monkeypatch):
        from hybridock_pep.toolpath import SIDECAR_ENV

        envs = tmp_path / "miniforge3" / "envs"
        prefix = envs / "score-env"
        (prefix / "bin").mkdir(parents=True)
        exe = _make_exe(envs / SIDECAR_ENV / "bin", "obabel")
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setattr(sys, "prefix", str(prefix))

        assert which("obabel") == str(exe)

    def test_own_env_wins_over_the_sidecar(self, tmp_path, monkeypatch):
        from hybridock_pep.toolpath import SIDECAR_ENV

        envs = tmp_path / "miniforge3" / "envs"
        prefix = envs / "score-env"
        mine = _make_exe(prefix / "bin", "obabel")
        _make_exe(envs / SIDECAR_ENV / "bin", "obabel")
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setattr(sys, "prefix", str(prefix))

        assert which("obabel") == str(mine)

    def test_no_sidecar_lookup_outside_a_conda_layout(self, tmp_path, monkeypatch):
        """sys.prefix must actually look like <base>/envs/<name> before we go
        guessing at sibling directories — a venv or system Python must not."""
        prefix = tmp_path / "venv"
        (prefix / "bin").mkdir(parents=True)
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setattr(sys, "prefix", str(prefix))

        assert which("obabel") is None

    def test_setup_environment_agrees_on_the_env_name(self):
        """The installer creates it; the pipeline looks for it. One name."""
        import importlib.util

        from hybridock_pep.toolpath import SIDECAR_ENV

        repo = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            "_setup_env_sidecar", repo / "scripts" / "setup_environment.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        assert mod.SIDECAR_ENV == SIDECAR_ENV
