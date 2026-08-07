"""uninstall.sh must never delete anything without explicit confirmation —
these tests only exercise the safe paths (--help, syntax, guard rails, and
declining the prompt). Nothing here is allowed to touch a real conda
environment or delete a real file; a test suite that could nuke the
environment it's running in would be worse than no test at all.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
UNINSTALL_SH = REPO_ROOT / "uninstall.sh"


def _bash_exe() -> str:
    """Locate a real POSIX bash to run uninstall.sh's own tests against.

    On native Windows CI runners, plain "bash" on $PATH can resolve to
    C:\\Windows\\System32\\bash.exe — the WSL launcher stub, not a real shell —
    depending on PATH order. With no WSL distro installed on that runner (it's
    not needed; conda-platforms' Windows job never uses WSL), that stub just
    prints "Windows Subsystem for Linux has no installed distributions" and
    exits 1, which uninstall.sh's own syntax has nothing to do with. Prefer Git
    for Windows' real bash (always present alongside GitHub Actions' windows-*
    runners). Same fix as test_install_script.py's _bash_exe() — this file
    just never got it when it was added later.
    """
    if os.name == "nt":
        for candidate in (
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
        ):
            if os.path.isfile(candidate):
                return candidate
    return "bash"


def _run(args, **kwargs):
    return subprocess.run(
        [_bash_exe(), str(UNINSTALL_SH), *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30, **kwargs,
    )


def test_uninstall_script_is_syntactically_valid():
    result = subprocess.run(
        [_bash_exe(), "-n", str(UNINSTALL_SH)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_uninstall_script_is_executable():
    """Windows has no POSIX executable bit, so st_mode's owner-execute bit is
    not a meaningful signal there — install.sh (this script's sibling, same
    git history) is real-world proof git preserves the +x bit through a
    Windows checkout without needing it re-asserted per-file."""
    if os.name == "nt":
        pytest.skip("no POSIX executable bit on Windows")
    assert UNINSTALL_SH.stat().st_mode & 0o111, "uninstall.sh must be chmod +x, same as install.sh"


def test_help_exits_zero_and_documents_every_flag():
    result = _run(["--help"])
    assert result.returncode == 0
    for flag in ("--all", "--envs-only", "--yes"):
        assert flag in result.stdout, f"{flag} is not documented in --help"


def test_help_explains_conda_itself_is_left_alone():
    """The single most important safety property of this script: it must
    never silently nuke someone's whole conda install. That has to be
    documented, not just true by accident of the implementation."""
    result = _run(["--help"])
    assert "miniforge3" in result.stdout.lower() or "conda install" in result.stdout.lower()


def test_unknown_flag_is_rejected():
    result = _run(["--this-flag-does-not-exist"])
    assert result.returncode != 0


def test_all_and_envs_only_are_mutually_exclusive():
    result = _run(["--all", "--envs-only"])
    assert result.returncode != 0


def test_declining_the_prompt_removes_nothing_and_exits_zero():
    """The default (no --yes) path must stop and ask before doing anything
    destructive — answering "n" must be a clean, harmless no-op."""
    result = _run([], input="n\n")
    assert result.returncode == 0
    assert "Aborted" in result.stdout

    result = _run([], input="\n")  # bare Enter -> the [y/N] default is No
    assert result.returncode == 0
    assert "Aborted" in result.stdout
