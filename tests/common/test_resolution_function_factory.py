"""Tests for the resolution function factory"""

import pytest

from MDMC.common.resolution_function_factory import ResolutionFunctionFactory
from MDMC.common.resolution_functions import *


@pytest.mark.parametrize('function, expected',
                         [('gaussian', gaussian),
                          ('lorentzian', lorentzian)])
def test_resolution_function_factory(function, expected):
    """
    Tests that the resolution function factory correctly creates function objects.
    """

    rff = ResolutionFunctionFactory()
    assert rff.create_instance(function) == expected


def test_resolution_function_factory_error():
    """
    Tests that an error is given when an unrecognised function is given.
    """

    rff = ResolutionFunctionFactory()
    with pytest.raises(NotImplementedError):
        rff.create_instance('fake_function')
