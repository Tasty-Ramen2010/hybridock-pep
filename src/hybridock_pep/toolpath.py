"""Locate external tools, including ones installed into our own conda env.

`shutil.which` searches `$PATH` and nothing else. That is the wrong scope here:
score-env.yml installs meeko, autogrid4 and openbabel *into score-env*, and the
console script the user runs lives in that same `score-env/bin`. Running it by
absolute path — `~/miniforge3/envs/score-env/bin/hybridock-pep`, which is what
`./install.sh`, the TUI, `ssh host hybridock-pep ...` and every subprocess in
this package do — leaves `score-env/bin` off `$PATH` entirely. Every tool the
installer just placed there is then invisible, and prep falls through to a
fallback that is not installed.

That failure looks like a missing dependency and is not one. Measured on a DGX
Spark: `mk_prepare_receptor.py` sitting in `score-env/bin` while receptor prep
reported "no receptor-prep tool found" and died on the obabel fallback.

So: `$PATH` first (an explicitly activated environment or a system install
should still win), then the running interpreter's own `bin/`.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

__all__ = ["SIDECAR_ENV", "which"]

#: Sibling conda env for tools that cannot be installed into score-env itself.
#: openbabel is the case that forced this: conda-forge builds it for
#: linux-aarch64 only against Python 3.9, so it cannot enter a 3.11 env, and
#: without it ARM Linux has no PDBQT converter at all. It installs cleanly into
#: an env of its own, and `obabel` is a binary — the Python version it was
#: built against is irrelevant to us. setup_environment.py creates this.
SIDECAR_ENV = "hdp-tools"


def _bindir(prefix: Path) -> Path:
    return prefix / ("Scripts" if os.name == "nt" else "bin")


def which(name: str) -> str | None:
    """Return the path to `name`, or None.

    Search order:
      1. `$PATH` — an explicitly activated env or system install wins.
      2. `sys.prefix/bin` — the env this interpreter is running from.
      3. `<conda base>/envs/hdp-tools/bin` — the sidecar env.

    Args:
        name: Executable name, e.g. "mk_prepare_receptor.py".

    Returns:
        Absolute path as a string, or None if not found anywhere.
    """
    found = shutil.which(name)
    if found is not None:
        return found

    # shutil.which() on Windows only matches names ending in a PATHEXT-registered
    # extension (.exe, .bat, ...). A script that's genuinely on $PATH verbatim with
    # no such extension - e.g. meeko's mk_prepare_receptor.py, which conda installs
    # as a bare .py file, no .exe wrapper - is invisible to it even though the file
    # is real and readable. Check PATH directories directly for an exact match too,
    # matching the same is_file()-based check already used for the env fallbacks
    # below (execute-bit checks don't mean much for a .py file on Windows anyway).
    if os.name == "nt":
        for directory in os.environ.get("PATH", "").split(os.pathsep):
            if not directory:
                continue
            candidate = Path(directory) / name
            if candidate.is_file():
                return str(candidate)

    prefix = Path(sys.prefix)
    candidate = _bindir(prefix) / name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)

    # sys.prefix is <base>/envs/<name>, so the sidecar is a sibling.
    if prefix.parent.name == "envs":
        candidate = _bindir(prefix.parent / SIDECAR_ENV) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None
