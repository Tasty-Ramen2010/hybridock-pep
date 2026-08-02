"""Emergency stop: does it actually kill everything the run started?

`hybridock-pep dock` launches the rapidock environment's python as a child to
do GPU sampling. Killing only the process the UI spawned leaves that child
alive — still holding the GPU, still writing into the output directory — while
the UI reports the run as stopped. That is the worst possible outcome for a
button whose entire purpose is "make it stop now".

So the run is started in its own process group and the whole group is
signalled. These tests use a throwaway shell that spawns its own child, which
reproduces the parent/child shape exactly without needing a GPU, a model, or a
minute of runtime.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

from hybridock_pep.ui.tui import terminate_process_tree

pytestmark = pytest.mark.skipif(sys.platform == "win32",
                                reason="POSIX process groups")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def parent_with_child():
    """A process in its own session that has spawned a long-lived child."""
    proc = subprocess.Popen(
        ["/bin/sh", "-c", "sleep 300 & echo $!; wait"],
        stdout=subprocess.PIPE, text=True, start_new_session=True,
    )
    child_pid = int(proc.stdout.readline().strip())
    assert _alive(child_pid)
    yield proc, child_pid
    # Belt and braces: never leak a sleeper if an assertion failed mid-test.
    for pid in (proc.pid, child_pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


class TestTerminateProcessTree:
    def test_kills_the_parent(self, parent_with_child):
        proc, _child = parent_with_child
        assert terminate_process_tree(proc, grace=2.0) is True
        assert proc.poll() is not None

    def test_kills_the_child_too(self, parent_with_child):
        """The whole point: an orphaned sampling child would keep using the GPU."""
        proc, child_pid = parent_with_child
        terminate_process_tree(proc, grace=2.0)
        assert _wait_gone(child_pid), (
            f"child {child_pid} survived the stop — it would still be running "
            "after the UI said the run was stopped"
        )

    def test_escalates_to_sigkill_when_sigterm_is_ignored(self):
        """A run wedged in a native call must still die."""
        proc = subprocess.Popen(
            [sys.executable, "-c",
             "import signal,time\n"
             "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
             "print('ready', flush=True)\n"
             "time.sleep(300)\n"],
            stdout=subprocess.PIPE, text=True, start_new_session=True,
        )
        try:
            assert proc.stdout.readline().strip() == "ready"
            assert terminate_process_tree(proc, grace=1.0) is True
        finally:
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def test_already_dead_process_is_fine(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        assert terminate_process_tree(proc) is True

    def test_none_is_fine(self):
        """Stop pressed before the subprocess was spawned."""
        assert terminate_process_tree(None) is True

    def test_never_raises_on_a_broken_process_object(self):
        """A stop button that can throw is not a stop button."""
        class Broken:
            pid = -12345

            def poll(self):
                return None

            def send_signal(self, sig):
                raise OSError("nope")

        terminate_process_tree(Broken(), grace=0.2)
