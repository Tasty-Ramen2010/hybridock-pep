"""Local web UI for HybriDock-Pep (``hybridock-pep serve``).

The browser front-end lives in ``index.html`` + ``static/``; :mod:`.server` is a
stdlib-only HTTP layer that hands the same commands to the CLI that the terminal
UI does. No web framework, so ``serve`` works in any environment that can already
run the tool.
"""

from __future__ import annotations

from .server import serve

__all__ = ["serve"]
