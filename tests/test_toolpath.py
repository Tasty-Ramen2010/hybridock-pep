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
