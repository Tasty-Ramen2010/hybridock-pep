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

__all__ = ["which"]


def which(name: str) -> str | None:
    """Return the path to `name`, or None.

    Checks `$PATH` first, then `sys.prefix/bin` (`Scripts` on Windows) — the
    environment this interpreter is running from.

    Args:
        name: Executable name, e.g. "mk_prepare_receptor.py".

    Returns:
        Absolute path as a string, or None if not found anywhere.
    """
    found = shutil.which(name)
    if found is not None:
        return found

    bindir = Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")
    candidate = bindir / name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None
