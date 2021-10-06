"""Tests for the resolution window factory"""

import pytest

from MDMC.trajectory_analysis.sqw_resolution_windows.resolution_window_factory import ResolutionWindowFactory
from MDMC.trajectory_analysis.sqw_resolution_windows.resolution_windows import *


@pytest.mark.parametrize('function, expected',
                         [('gaussian', gaussian_window),
                          ('lorentzian', lorentzian_window)])
def test_resolution_window_factory(function, expected):
    """
    Tests that the resolution function factory correctly creates function objects.
    """

    rwf = ResolutionWindowFactory()
    assert rwf.create_instance(function) == expected


def test_resolution_window_factory_error():
    """
    Tests that an error is given when an unrecognised function is provided as a parameter.
    """

    rwf = ResolutionWindowFactory()
    with pytest.raises(NotImplementedError):
        rwf.create_instance('fake_function')
