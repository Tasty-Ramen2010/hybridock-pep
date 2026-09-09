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


@pytest.fixture(autouse=True)
def _clear_openmm_platform_cache():
    """Isolate the process-global OpenMM platform probe between tests.

    ``hybridock_pep.hardware.openmm_platform`` caches its probe for the life of
    the process — a working backend does not change mid-run, and re-probing a
    broken one strands device memory. That cache is shared by every test that
    mocks ``openmm``, so without this the first such test decides the answer for
    all the rest.
    """
    from hybridock_pep.hardware import reset_platform_cache

    reset_platform_cache()
    yield
    reset_platform_cache()
