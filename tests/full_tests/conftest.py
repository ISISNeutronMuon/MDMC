"""
Code for custom --argon option.

Tests marked with @pytest.mark.argon will run only if --argon is invoked
in the pytest command line options.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--argon",
        action='store_true',
        default=False,
        help="Runs full Argon system test.",
    )


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers", "argon: mark test to run only when --argon option is invoked."
    )


def pytest_runtest_setup(item):
    if item.iter_markers(name="argon"):
        if not item.config.getoption("--argon"):
            pytest.skip("test only runs if --argon is invoked")