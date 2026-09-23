"""Finding a real shell to run this repo's .sh scripts from the test suite.

Windows-only problem, but it has cost us a red CI for weeks, so it lives in one
place now instead of being re-solved per test file.
"""

from __future__ import annotations

import os
import shutil

#: Git for Windows ships a genuine bash and is always present on GitHub's
#: windows-* runners (the workflow uses `shell: bash` steps, which is that bash).
_GIT_BASH = (
    r"C:\Program Files\Git\bin\bash.exe",
    r"C:\Program Files (x86)\Git\bin\bash.exe",
)


def bash_exe() -> str:
    """Return a bash that can actually run a POSIX script.

    On Windows, plain ``bash`` on PATH usually resolves to
    ``C:\\Windows\\System32\\bash.exe`` — the WSL *launcher*, not a shell. With no
    WSL distro installed (CI runners have none, and don't need one) that stub
    prints "Windows Subsystem for Linux has no installed distributions" as
    UTF-16 and exits 1, so every test that shells out to a script fails for a
    reason that has nothing to do with the script. Prefer Git for Windows' real
    bash, and fall back to whatever PATH offers elsewhere.

    Returns:
        Path to a bash executable, or the bare name ``"bash"`` on POSIX.
    """
    if os.name != "nt":
        return "bash"
    for candidate in _GIT_BASH:
        if os.path.isfile(candidate):
            return candidate
    found = shutil.which("bash")
    if found and "system32" not in found.lower():
        return found
    return "bash"


def have_real_bash() -> bool:
    """True when :func:`bash_exe` found something that is not the WSL stub.

    Used to skip script tests on a Windows box with no Git Bash, rather than
    reporting a shell-resolution problem as a failure of the script under test.
    """
    exe = bash_exe()
    if os.name != "nt":
        return shutil.which("bash") is not None
    return exe != "bash" or bool(
        (shutil.which("bash") or "") and "system32" not in (shutil.which("bash") or "").lower()
    )
