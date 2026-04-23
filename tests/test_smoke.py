"""Smoke tests for the vega-mir package."""

import vega_mir


def test_version() -> None:
    assert vega_mir.__version__ == "0.0.1"


def test_import() -> None:
    assert vega_mir is not None
