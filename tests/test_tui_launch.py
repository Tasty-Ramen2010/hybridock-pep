"""Does the full-screen UI actually come up?

Every other TUI test imports the module and calls functions. None of them can
catch a prompt_toolkit *layout* error, because layouts are only validated once
a terminal is attached — and that is exactly how the first-run welcome screen
shipped broken during development:

    full-screen UI unavailable: Invalid value. Window does not appear in the
    layout: Window(content=<BufferControl ...>); falling back to --cli

Focus was left on a form input while the welcome view was on screen.
prompt_toolkit rejected the layout, the app silently degraded to the plain
`--cli` wizard, and the entire guided-tour feature was invisible. The unit
tests all passed.

So this launches the real binary under a real pty. Marked `slow` (it spawns a
process and waits on terminal I/O); run with `pytest -m slow`.
"""

from __future__ import annotations

import os
import re
import shutil
import sys

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(sys.platform == "win32", reason="pty is POSIX-only"),
    pytest.mark.skipif(shutil.which("hybridock-tui") is None,
                       reason="hybridock-tui not installed"),
]

ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b[()][AB012]")


def _launch(home: str, keys: bytes = b"", settle: float = 2.5) -> str:
    """Run the TUI in a pty, optionally send keys, return the de-ANSI'd screen."""
    import fcntl
    import pty
    import select
    import struct
    import termios
    import time

    env = dict(os.environ, TERM="xterm-256color", HOME=home)
    pid, fd = pty.fork()
    if pid == 0:  # child
        os.execvpe("hybridock-tui", ["hybridock-tui"], env)

    # The child sizes itself from the pty; LINES/COLUMNS in the env are ignored.
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 44, 100, 0, 0))

    buf = b""
    try:
        if keys:
            time.sleep(settle)
            os.write(fd, keys)
        deadline = time.time() + 10
        while time.time() < deadline:
            r, _, _ = select.select([fd], [], [], 0.4)
            if not r:
                continue
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
        os.write(fd, b"\x11")  # Ctrl-Q
        time.sleep(0.4)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass

    return ANSI.sub("", buf.decode("utf-8", "replace"))


@pytest.fixture
def fresh_home(tmp_path):
    """A HOME with no welcome marker — i.e. a brand-new user."""
    return str(tmp_path / "fresh")


@pytest.fixture
def returning_home(tmp_path):
    """A HOME whose user has already dismissed the tour."""
    home = tmp_path / "returning"
    (home / ".hybridock-pep").mkdir(parents=True)
    (home / ".hybridock-pep" / "ui_welcomed").write_text("1\n")
    return str(home)


def test_fullscreen_ui_does_not_fall_back_to_cli(fresh_home):
    """The regression that motivated this file."""
    screen = _launch(fresh_home)
    assert "full-screen UI unavailable" not in screen, (
        "the full-screen layout was rejected and the app degraded to --cli"
    )


def test_first_run_shows_the_welcome_tour(fresh_home):
    screen = _launch(fresh_home)
    assert "WELCOME" in screen
    assert "WALKTHROUGH" in screen


def test_returning_user_gets_the_form_not_the_tour(returning_home):
    screen = _launch(returning_home)
    assert "Peptide sequence" in screen
    assert "THE FIVE-STEP WALKTHROUGH" not in screen


def _topics_visible(screen: str) -> int:
    """How many help-menu topic titles are on screen.

    Asserted instead of the menu header because the header is one long line:
    prompt_toolkit repaints incrementally, so a captured frame can split it
    mid-string even though the menu is fully rendered. Short topic titles
    survive that; the header does not.
    """
    from hybridock_pep.ui import help_content
    return sum(1 for _key, title in help_content.topic_titles() if title in screen)


def test_help_opens_and_switches_topic(returning_home):
    """Ctrl-G opens help; a number key selects a topic."""
    screen = _launch(returning_home, keys=b"\x07" + b"5")
    assert _topics_visible(screen) >= 3, "help menu did not render"
    assert "SELECTIVITY" in screen.upper(), "topic 5 body was not shown"


def test_help_is_reachable_for_a_first_time_user_too(fresh_home):
    """A new user must be able to get help straight from the tour."""
    screen = _launch(fresh_home, keys=b"\x07")
    assert _topics_visible(screen) >= 3
