"""Resolve packaged data files whether installed as a wheel or run from a source checkout.

`Path(__file__).resolve().parents[N] / "data" / ...` only works in an editable
install (`pip install -e .`), where `src/hybridock_pep/` has a sibling repo-root
`data/` directory. A real (non-editable) wheel install has no such sibling — the
files simply aren't there. This module is the single place that knows both
layouts: it prefers the packaged copy under `hybridock_pep/data/` (shipped in
the wheel via `[tool.setuptools.package-data]`), falling back to the source
checkout's repo-root `data/` (unchanged behavior for developers and tests
running against a git clone).
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path

_DEV_REPO_ROOT = Path(__file__).resolve().parents[2]


def data_file(relpath: str) -> Path:
    """Return the path to a shipped data file, packaged or dev-checkout.

    Args:
        relpath: Path relative to the ``data/`` directory, e.g.
            ``"affinity_ai_nofix.joblib"`` or ``"pdbs/1YCR_mdm2.pdb"``.

    Returns:
        The resolved path. Callers that need existence guarantees should still
        check ``.exists()`` — this function does not raise if neither location
        has the file, matching the historical (silent) behavior of the
        hardcoded paths it replaces.
    """
    packaged = importlib.resources.files("hybridock_pep") / "data" / relpath
    if packaged.is_file():
        return Path(str(packaged))
    return _DEV_REPO_ROOT / "data" / relpath
