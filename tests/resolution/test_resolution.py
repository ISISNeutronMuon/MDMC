"""
Contains tests for the Resolution classes.
Note that testing of applying the functions, i.e.
validation is done in tests/system_tests/observables.
"""

import pytest

from inspect import isfunction

import MDMC.resolution as res
from tests.test_data import data

# we get the list of resolution functions from the module and use it to automatically create our `parametrize` cases.
# This means that new functions will automatically be added to the tests when implemented.
resdict = res.ResolutionFactory.registry
resparms = list(resdict.values())


@pytest.mark.parametrize('resolution', resparms)
def test_resolution_initialise(resolution):
    """
    Tests that all resolution functions can be instantiated correctly.
    """
    if resolution == res.NullResolution:  # test null resolution has no attributes
        resfunc = resolution(0)
        attributes = [x for x in dir(resfunc) if '_' not in x]
        assert attributes == ['apply']
    elif resolution == res.FileResolution:  # test file resolution has a file name and an array
        resfunc = resolution(data.RESOLUTION_DATA['LAMPSQw'],
                             'SQw', 'LAMPSQw', 1055.8303421611213)
        assert resfunc.file_name == data.RESOLUTION_DATA['LAMPSQw']
        assert isfunction(resfunc.resolution_function)
    else:  # else, resolution is numeric
        resfunc = resolution(84.0)
        assert resfunc.e_res == 0.084  # unit conversion turns FWHM from ueV to meV
