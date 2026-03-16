"""Shared pytest configuration and fixtures."""

from __future__ import annotations


def pytest_collection_modifyitems(config, items):
    """Skip integration and slow tests unless explicitly requested via -m."""
    # If the user passed an explicit -m expression, respect it as-is.
    if config.getoption("-m"):
        return

    import pytest

    skip_integration = pytest.mark.skip(reason="integration test — run with: pytest -m integration")
    skip_slow = pytest.mark.skip(reason="slow test — run with: pytest -m slow")

    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
        elif "integration" in item.keywords:
            item.add_marker(skip_integration)
