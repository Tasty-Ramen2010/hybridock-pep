"""uninstall.sh must never delete anything without explicit confirmation —
these tests only exercise the safe paths (--help, syntax, guard rails, and
declining the prompt). Nothing here is allowed to touch a real conda
environment or delete a real file; a test suite that could nuke the
environment it's running in would be worse than no test at all.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UNINSTALL_SH = REPO_ROOT / "uninstall.sh"


def _run(args, **kwargs):
    return subprocess.run(
        ["bash", str(UNINSTALL_SH), *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30, **kwargs,
    )


def test_uninstall_script_is_syntactically_valid():
    result = subprocess.run(["bash", "-n", str(UNINSTALL_SH)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_uninstall_script_is_executable():
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
