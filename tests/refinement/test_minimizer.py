"""Tests the Minimizer class and subclasses
"""

from unittest.mock import patch

import numpy as np
import pytest

from MDMC.refinement import minimizer


class MockParameter:

    def __init__(self, name, value):

        self.name = name
        self.value = value
        self.fixed = False


@pytest.fixture
def parameters():

    """
    Returns
    -------
    list
        A `list` of ``MockParameter`` objects with a variety of `name` and
        `value` attributes
    """

    return([MockParameter('charge', 1.),
            MockParameter('charge', .5),
            MockParameter('sigma', 3.3),
            MockParameter('epsilon', .2),
            MockParameter('potential_strength', 1234.),
            MockParameter('equilibrium_state', 1.2),
            MockParameter('A', 1.),
            MockParameter('B', 2.),
            MockParameter('C', 3.)])


@patch.multiple('MDMC.refinement.minimizer.Minimizer', __abstractmethods__=set())
def test_minimizer_init(parameters):

    """
    Test initializing Minimizer
    """

    # Ignore pylint error as abstract class is mocked
    # pylint: disable=abstract-class-instantiated
    minim = minimizer.Minimizer(1, parameters)
    assert np.all(minim.params == np.array(parameters))


@patch.multiple('MDMC.refinement.minimizer.Minimizer', __abstractmethods__=set())
def test_minimizer_init_invalid_params(parameters):

    """
    Test initializing Minimizer with fixed parameters, which should raise a
    ValueError
    """

    # Ignore pylint error as abstract class is mocked
    # pylint: disable=abstract-class-instantiated
    fixed_parameter = MockParameter('charge', 1.)
    fixed_parameter.fixed = True

    with pytest.raises(ValueError):
        minimizer.Minimizer(1, parameters + [fixed_parameter])


@pytest.mark.parametrize('p_slice, columns',
                         [([0, 4, 1],
                           ['charge', 'charge', 'sigma', 'epsilon']),
                          ([0, 9, 2],
                           ['charge', 'sigma', 'potential_strength', 'A', 'C'])
                         ])

def test_MMC_history_columns(parameters, p_slice, columns):

    mmc = minimizer.MMC(1, parameters[slice(*p_slice)])
    assert mmc.history_columns == ['FoM', 'Old FoM', 'Change state'] + columns
