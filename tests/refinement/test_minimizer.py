"""Tests the Minimizer class and subclasses
"""

from tempfile import NamedTemporaryFile
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


@patch.multiple('MDMC.refinement.minimizer.Minimizer', __abstractmethods__=set()
               )
def test_minimizer_init(parameters):

    """
    Test initializing ``Minimizer``
    """

    # Ignore pylint error as abstract class is mocked
    # pylint: disable=abstract-class-instantiated
    minim = minimizer.Minimizer(1, parameters)
    assert np.all(minim.params == np.array(parameters))


@patch.multiple('MDMC.refinement.minimizer.Minimizer', __abstractmethods__=set()
               )
def test_minimizer_init_invalid_params(parameters):

    """
    Test initializing ``Minimizer`` with fixed parameters, which should raise a
    `ValueError`
    """

    # Ignore pylint error as abstract class is mocked
    # pylint: disable=abstract-class-instantiated
    fixed_parameter = MockParameter('charge', 1.)
    fixed_parameter.fixed = True

    with pytest.raises(ValueError):
        minimizer.Minimizer(1, parameters + [fixed_parameter])


@patch.multiple('MDMC.refinement.minimizer.Minimizer', __abstractmethods__=set())
def test_minimizer_write_history(parameters):

    """
    Test history csv output of ``Minimizer``
    """

    class MockMinimizer(minimizer.Minimizer):

        @property
        def history_columns(self):

            return ['A', 'B', 'C']

    # Ignore pylint error as abstract class is mocked
    # pylint: disable=abstract-class-instantiated
    minim = MockMinimizer(1, parameters)
    minim._history = [[10., 20., 30.],
                      ['Accepted', 'Rejected', 'Accepted'],
                      [3., 4., 5.874958734958]]
    tfile = NamedTemporaryFile()
    minim.write_history(tfile.name)
    lines = tfile.readlines()
    assert lines == [b',A,B,C\n',
                     b'0,10.0,20.0,30.0\n',
                     b'1,Accepted,Rejected,Accepted\n',
                     b'2,3.0,4.0,5.874958734958\n']

@pytest.mark.parametrize('p_slice, columns',
                         [([0, 4, 1],
                           ['charge', 'charge', 'sigma', 'epsilon']),
                          ([0, 9, 2],
                           ['charge', 'sigma', 'potential_strength', 'A', 'C'])
                         ])
def test_mmc_history_columns(parameters, p_slice, columns):

    """
    Tests that the history columns for the ``MMC`` minimizer are as expected,
    including the names of the ``Parameter`` objects which are refined
    """

    mmc = minimizer.MMC(1, parameters[slice(*p_slice)])
    assert mmc.history_columns == ['FoM', 'Old FoM', 'Change state'] + columns
