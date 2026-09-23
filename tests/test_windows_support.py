"""Windows portability.

These run on every platform: they check the *decisions* the code makes about
Windows (which bash, which process-group flag, which conda channels), not the
behaviour of Windows itself, so a Linux CI job still catches a regression that
would only bite a Windows user.

Background: the Windows CI job was red from August to 2026-09-23 because tests
shelled out to ``bash``, which on Windows resolves to the WSL launcher stub
rather than a shell.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest import mock

import pytest
import yaml

from hybridock_pep.web.server import _process_group_kwargs
from tests.shell import bash_exe, have_real_bash

REPO = Path(__file__).resolve().parent.parent
GIT_BASH = r"C:\Program Files\Git\bin\bash.exe"
WSL_STUB = r"C:\Windows\System32\bash.exe"


class TestBashResolution:
    """The thing that was actually breaking CI."""

    def test_posix_uses_plain_bash(self):
        with mock.patch.object(os, "name", "posix"):
            assert bash_exe() == "bash"

    def test_windows_prefers_git_bash(self):
        with mock.patch.object(os, "name", "nt"), \
             mock.patch.object(os.path, "isfile", lambda p: p == GIT_BASH):
            assert bash_exe() == GIT_BASH

    def test_windows_refuses_the_wsl_launcher_stub(self):
        """System32\\bash.exe is the WSL launcher: it is not a shell.

        With no distro installed it prints "Windows Subsystem for Linux has no
        installed distributions" as UTF-16 and exits 1, which is what made ten
        unrelated script tests fail.
        """
        with mock.patch.object(os, "name", "nt"), \
             mock.patch.object(os.path, "isfile", lambda p: False), \
             mock.patch("shutil.which", lambda name: WSL_STUB):
            assert bash_exe() != WSL_STUB
            assert have_real_bash() is False

    def test_windows_accepts_some_other_real_bash_on_path(self):
        other = r"C:\tools\msys64\usr\bin\bash.exe"
        with mock.patch.object(os, "name", "nt"), \
             mock.patch.object(os.path, "isfile", lambda p: False), \
             mock.patch("shutil.which", lambda name: other):
            assert bash_exe() == other

    @pytest.mark.skipif(not have_real_bash(), reason="no POSIX bash here")
    def test_the_resolved_bash_actually_runs(self):
        out = subprocess.run([bash_exe(), "-c", "echo alive"],
                             capture_output=True, text=True, check=False)
        assert out.returncode == 0
        assert "alive" in out.stdout


class TestProcessGroups:
    """Stopping a run has to kill the GPU child too, on both platforms."""

    def test_posix_asks_for_a_new_session(self):
        with mock.patch.object(os, "name", "posix"):
            assert _process_group_kwargs() == {"start_new_session": True}

    def test_windows_asks_for_a_new_process_group(self):
        with mock.patch.object(os, "name", "nt"):
            kwargs = _process_group_kwargs()
        assert "creationflags" in kwargs
        assert "start_new_session" not in kwargs, (
            "Windows ignores start_new_session, so the stop button would only "
            "kill the parent and leave RAPiDock holding the GPU"
        )

    def test_terminate_uses_taskkill_on_windows(self):
        from hybridock_pep.ui import tui

        proc = mock.Mock()
        proc.poll.side_effect = [None, 0]
        proc.pid = 4321
        with mock.patch.object(os, "name", "nt"), \
             mock.patch.object(subprocess, "run") as run:
            assert tui.terminate_process_tree(proc) is True
        cmd = run.call_args[0][0]
        assert cmd[0] == "taskkill"
        assert "/T" in cmd and "/F" in cmd and "4321" in cmd


class TestCondaEnvironmentFiles:
    """`conda env create` has to work on a Windows box, not just a Linux one."""

    @pytest.mark.parametrize("name", ["score-env.yml", "rapidock-env.yml",
                                      "rapidock-env-macos.yml", "diffpepdock-env.yml"])
    def test_no_defaults_channel(self, name):
        """`defaults` pulls in repo.anaconda.com/pkgs/msys2 on Windows.

        Conda's Terms-of-Service gate then aborts the whole solve with
        CondaToSNonInteractiveError before installing anything. Every package in
        these files is conda-forge-pinned anyway, so `defaults` buys nothing.
        """
        spec = yaml.safe_load((REPO / "envs" / name).read_text())
        assert "defaults" not in (spec.get("channels") or []), (
            f"{name} lists the defaults channel; that breaks conda env create on Windows"
        )

    def test_conda_forge_is_still_there(self):
        spec = yaml.safe_load((REPO / "envs/score-env.yml").read_text())
        assert "conda-forge" in spec["channels"]
