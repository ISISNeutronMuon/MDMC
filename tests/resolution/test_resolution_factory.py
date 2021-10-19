"""
Contains tests for the ResolutionFactory factory pattern.
"""

import pytest

import MDMC.resolution as res
from tests.test_data import data

# turns the dict of resolution functions into a list of tuples for case parameterisation;
# first entry is the user string input to get the resolution type, the second is the function.
# this means that new resolution types are automatically added to the parameterisation when they are implemented.
rf = res.ResolutionFactory()
resdict = rf.resolutions
reslist = [(str(list(resdict.keys())[i])[:-10].lower(), list(resdict.values())[i]) for i in range(0, len(resdict))]


@pytest.mark.parametrize('resolution, expected', reslist)
def test_resolution_factory(resolution, expected):
    """
    Tests that input of working functions to the resolution factory gives the correct result.
    """
    resolution_factory = res.ResolutionFactory()

    if resolution == 'file':
        resolution_function = rf.create_instance({resolution: data.RESOLUTION_DATA['LAMPSQw']},
                                        'SQw', 'LAMPSQw', 1055.8303421611213)
    else:
        resolution_function = rf.create_instance({resolution: 84})

    assert type(resolution_function) == expected


@pytest.mark.parametrize('resolution, expected, test_warning',
                         [(84, res.GaussianResolution, True),
                          (84.0, res.GaussianResolution, True),
                          (data.RESOLUTION_DATA['LAMPSQw'], res.FileResolution, False),
                          (None, res.NullResolution, False),
                          ({'lorentzian': 84, 'gaussian': 85}, res.LorentzianResolution, True)])
def test_resolution_factory_input_handling(resolution, expected, test_warning):
    """
    Tests that when resolution is given as accepted methods other than a one-line dictionary,
    that it is handled correctly. That is:
    - if a float or int, assume Gaussian, convert to dict and give a warning
    - if a string, assume file and convert to dict
    - if None, change to null pattern
    - if a multi-line dictionary, take only the first line entered
    """

    if test_warning:  # if we are testing a warning is given, i.e. resolution is numeric
        with pytest.warns(SyntaxWarning):
            # don't bother adding FileResolution args as no file will be in this part of the subroutine
            resolution_function = rf.create_instance(resolution)
    else:
        resolution_function = rf.create_instance(resolution, 'SQw', 'LAMPSQw', 1055.8303421611213)

    assert type(resolution_function) == expected
