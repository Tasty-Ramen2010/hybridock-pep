"""pytest configuration: auto-skip tests marked @pytest.mark.slow unless -m slow is given."""
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    marker_expr = config.getoption("-m", default="")
    if "slow" in str(marker_expr):
        return
    skip_slow = pytest.mark.skip(reason="slow test — run with: pytest -m slow")
    for item in items:
        if item.get_closest_marker("slow"):
            item.add_marker(skip_slow)
