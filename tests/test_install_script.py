"""install.sh regressions that only show up on a machine that is *not* freshly
set up the way the author's machine was.

Both bugs covered here were found by running the documented install on a second
machine (an aarch64 DGX Spark) over a plain `ssh host ./install.sh`:

1. `command -v conda` fails in a non-interactive shell even when conda is
   installed, so the script tried to install Miniforge over an existing
   ~/miniforge3, the Miniforge installer refused, and `set -euo pipefail`
   killed the install with "File or directory already exists".
2. A submodule gitlink with no .gitmodules entry made
   `git submodule update --init --recursive` exit 128 *before* RAPiDock was
   fetched, so a plain (non `--recurse-submodules`) clone could never install.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "install.sh"


def test_install_script_is_syntactically_valid():
    result = subprocess.run(
        ["bash", "-n", str(INSTALL_SH)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


class TestSubmoduleGitlinks:
    """Every gitlink in the tree must have a .gitmodules entry.

    `git submodule update --init --recursive` is a hard error (exit 128) on an
    orphan gitlink, and install.sh runs it under `set -e`. Because submodules
    are processed in path order, an orphan sorting before third_party/RAPiDock
    aborts the install with the sampler never downloaded.
    """

    def test_no_orphan_gitlinks(self):
        git = shutil.which("git")
        if git is None or not (REPO_ROOT / ".git").exists():
            pytest.skip("not a git checkout")
        listing = subprocess.run(
            [git, "ls-files", "-s"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout
        gitlinks = {
            line.split("\t", 1)[1]
            for line in listing.splitlines()
            if line.startswith("160000")
        }
        declared = subprocess.run(
            [git, "config", "-f", ".gitmodules", "--get-regexp", r"^submodule\..*\.path$"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        ).stdout
        declared_paths = {ln.split(" ", 1)[1] for ln in declared.splitlines() if " " in ln}
        orphans = gitlinks - declared_paths
        assert not orphans, (
            f"gitlink(s) with no .gitmodules entry: {sorted(orphans)} — "
            "`git submodule update --init --recursive` will exit 128 and install.sh "
            "dies with it"
        )


class TestExistingCondaIsFound:
    """An installed-but-not-on-PATH conda must be discovered, not reinstalled."""

    def _run_step_one(self, tmp_path: Path, *, with_conda_install: bool) -> subprocess.CompletedProcess:
        """Run install.sh with a PATH that has no conda and stubs for anything
        that would otherwise touch the network or create environments.

        The script is expected to fail *somewhere later* (the stubs are not real
        tools); the assertions are about what it did before that.
        """
        home = tmp_path / "home"
        home.mkdir()
        if with_conda_install:
            profile = home / "miniforge3" / "etc" / "profile.d"
            profile.mkdir(parents=True)
            (profile / "conda.sh").write_text("conda() { echo stub-conda; }\n")

        stubs = tmp_path / "stubs"
        stubs.mkdir()
        calls = tmp_path / "calls.log"
        for name in ("curl", "python3", "sha256sum"):
            stub = stubs / name
            stub.write_text(f'#!/bin/sh\necho "{name} $*" >> "{calls}"\nexit 1\n')
            stub.chmod(0o755)

        env = dict(os.environ)
        env["HOME"] = str(home)
        env["PATH"] = f"{stubs}:/usr/bin:/bin"
        proc = subprocess.run(
            ["bash", str(INSTALL_SH), "--no-ui"],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300,
        )
        proc.calls = calls.read_text() if calls.exists() else ""  # type: ignore[attr-defined]
        return proc

    def test_does_not_download_miniforge_over_an_existing_install(self, tmp_path):
        proc = self._run_step_one(tmp_path, with_conda_install=True)
        assert "conda found" in proc.stdout, proc.stdout[:2000]
        miniforge_fetches = [
            ln for ln in proc.calls.splitlines()  # type: ignore[attr-defined]
            if ln.startswith("curl") and "iniforge" in ln
        ]
        assert not miniforge_fetches, (
            "install.sh tried to install Miniforge on top of an existing conda: "
            f"{miniforge_fetches}"
        )

    def test_still_installs_miniforge_when_there_is_genuinely_no_conda(self, tmp_path):
        """The discovery shortcut must not disable the auto-install it guards."""
        if Path("/opt/conda/etc/profile.d/conda.sh").exists():
            pytest.skip("machine-wide /opt/conda would be discovered regardless of $HOME")
        proc = self._run_step_one(tmp_path, with_conda_install=False)
        assert "conda not found" in proc.stdout, proc.stdout[:2000]
        assert any(
            ln.startswith("curl") and "iniforge" in ln
            for ln in proc.calls.splitlines()  # type: ignore[attr-defined]
        ), f"no Miniforge download attempted; calls were:\n{proc.calls}"  # type: ignore[attr-defined]
