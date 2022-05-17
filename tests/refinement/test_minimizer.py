"""
Tests the Minimizer base class
"""

from tempfile import NamedTemporaryFile

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.gaussian_process.kernels import RBF

from MDMC.MD.parameters import Parameter, Parameters
from MDMC.refinement import minimizers
from MDMC.refinement.minimizers.minimizer_factory import MinimizerFactory
from MDMC.refinement.minimizers.minimizer_abs import Minimizer

pytestmark = pytest.mark.mpi


@pytest.fixture
def parameters():
    """
    Returns
    -------
    Parameters
        A `dict-like` of ``Parameter`` objects with a variety of `name` and
        `value` attributes
    """

    return Parameters([Parameter(name='A', value=1.),
                       Parameter(name='B', value=2.),
                       Parameter(name='C', value=3.),
                       Parameter(name='charge', value=1.),
                       Parameter(name='charge', value=.5),
                       Parameter(name='epsilon', value=.2),
                       Parameter(name='equilibrium_state', value=1.2),
                       Parameter(name='potential_strength', value=1234.),
                       Parameter(name='sigma', value=3.3)])


@patch.multiple(Minimizer, __abstractmethods__=set())
def test_minimizer_init(parameters):
    """
    Test initializing ``Minimizer``
    """
    # it's not worth parametrising a fixture just for this, so we use a loop
    cases = [parameters, list(parameters.values())]

    # Ignore pylint error as abstract class is mocked
    # pylint: disable=abstract-class-instantiated
    for parms in cases:
        minim = Minimizer(parms)
        assert np.all(minim.parameters == parameters)


@patch.multiple(Minimizer, __abstractmethods__=set())
def test_minimizer_init_invalid_parameters(parameters):
    """
    Test initializing ``Minimizer`` with fixed parameters, which should raise a
    `ValueError`
    """

    # Ignore pylint error as abstract class is mocked
    # pylint: disable=abstract-class-instantiated
    fixed_parameter = Parameter(name='fixed_parameter', value=1.)
    fixed_parameter.fixed = True
    parameters.append(fixed_parameter)

    with pytest.raises(ValueError):
        Minimizer(parameters)


@patch.multiple(Minimizer, __abstractmethods__=set())
def test_minimizer_write_history(parameters):
    """
    Test history csv output of ``Minimizer``
    """

    class MockMinimizer(Minimizer):

        @property
        def history_columns(self):
            return ['A', 'B', 'C']

    # Ignore pylint error as abstract class is mocked
    # pylint: disable=abstract-class-instantiated
    minim = MockMinimizer(parameters)
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
                           ['A', 'B', 'C', 'charge']),
                          ([0, 9, 2],
                           ['A', 'C', 'charge', 'equilibrium_state', 'sigma'])
                          ])
def test_minimizer_history_columns(parameters, p_slice, columns):
    """
    Tests that the history columns for the` minimizer are as expected,
    including the names of the ``Parameter`` objects which are refined
    """
    parameter_slice = Parameters(list(parameters.values())[slice(*p_slice)])

    for minimizer_name in MinimizerFactory.get_minimizer_names():
        minim = MinimizerFactory.create_minimizer(minimizer_name, parameter_slice)
        assert minim.history_columns == ['FoM', 'Change state'] + [list(columns)]


def mock_change_parameters(self, parameters):
    """
    Mock of minimizer.change_parameters which doubles each ``Parameter`` value
    """

    for p in parameters:
        parameters[p].value *= 2

@pytest.mark.skip(reason="This test fails due to parameter duplication being inconsistent. Will work again once ID system is implemented")
def test_minimizer_change_parameters(parameters):
    """
    Tests that the parameters change by the expected amount when given a mocked
    distribution which always returns 1.
    """

    def mock_distribution(low: float, high: float, size: int):
        return np.ones(size)

    expected_values = {p: 2 * parameters[p].value for p in parameters}
    for minimizer_name in MinimizerFactory.get_minimizer_names():
        minim = MinimizerFactory.create_minimizer(minimizer_name, parameters)
        minim.distribution = mock_distribution
        minim.change_parameters(minim.parameters)
        for p in minim.parameters:
            assert minim.parameters[p].value == expected_values[p]


def test_minimizer_change_constrained_parameter():
    """
    Tests that constrained parameters do not exceed their max/min values.
    """

    def mock_distribution(low: float, high: float, size: int):
        # For non-constrained parameters, this would result in the first being doubled and the
        # second set to 0
        return np.array([1., -1.])

    parameters = Parameters([Parameter(name='constraints', value=1., constraints=(0.5, 1.5)),
                             Parameter(name='constraints_2', value=1., constraints=(0.5, 1.5))])

    # Expect values to be set to the upper/lower limit
    expected_values = [1.5, 0.5]
    for minimizer_name in MinimizerFactory.get_minimizer_names():
        minim = MinimizerFactory.create_minimizer(minimizer_name, parameters)
        minim.distribution = mock_distribution
        minim.change_parameters(minim.parameters)
        assert [p.value for p in minim.parameters.values()] == expected_values


@pytest.mark.parametrize('FoM, FoM_old',
                         [(10., 20.),
                          (10., 20.),
                          (10., 20.),
                          (1.e+10, 1.e+10),
                          (0., 0.)])
def test_MMC_minimizer_change_state_FoM_le(monkeypatch, parameters, FoM, FoM_old):
    """
    Tests that the state always changes (i.e. returns True) given an FoM, old
    FoM, and MC norm, where the FoM is less than or equal to old FoM, and the
    return of ``np.random.random`` is 0.999
    """

    def mock_random():
        return 0.999999

    for minimizer_name in MinimizerFactory.get_minimizer_names():
        minim = MinimizerFactory.create_minimizer(minimizer_name, parameters, MC_norm=1.0)
        minim.FoM_old = FoM_old
        minim.FoM = FoM
        monkeypatch.setattr(np.random, 'random', mock_random)
        assert minim.change_state() is True


@pytest.mark.parametrize('FoM, FoM_old, MC_norm, change',
                         [(21., 20., 1., False),
                          (21., 20., 2., True),
                          (21., 20., 100., True),
                          (40., 20., 29., True),
                          (40., 20., 28., False),
                          (60., 40., 29., True),
                          (60., 40., 28., False)])
def test_minimizer_change_state_FoM_gt(monkeypatch, parameters, FoM, FoM_old,
                                       MC_norm, change):
    """
    Tests that the state changes correctly given an FoM, old FoM, and MC norm,
    where the FoM is greater than the old FoM, and the return of
    ``np.random.random`` is 0.5
    """

    def mock_random():
        return 0.5

    for minimizer_name in MinimizerFactory.get_minimizer_names():
        minim = MinimizerFactory.create_minimizer(minimizer_name, parameters, MC_norm=MC_norm)
        minim.FoM_old = FoM_old
        minim.FoM = FoM
        monkeypatch.setattr(np.random, 'random', mock_random)
        assert minim.change_state() == change


@pytest.mark.parametrize('mock_history, min_steps, expected',
                         [([[3, 'Accepted', 4], [2, 'Accepted', 3], [1, 'Accepted', 2]], None,
                           False),
                          ([[3, 'Accepted', 4], [2, 'Accepted', 3], [2, 'Accepted', 2]], None,
                           False),
                          ([[3, 'Accepted', 4], [2, 'Rejected', 3], [2, 'Accepted', 3]], None,
                           False),
                          ([[3, 'Accepted', 4], [2, 'Accepted', 3], [1, 'Accepted', 3]], None,
                           False),
                          ([[2, 'Accepted', 4], [2, 'Rejected', 4], [2, 'Rejected', 4]], None,
                           False),
                          ([[3, 'Accepted', 4], [2, 'Accepted', 3], [2, 'Accepted', 3]], 4, False),
                          ([[3, 'Accepted', 4], [2, 'Accepted', 3], [2, 'Accepted', 3]], None,
                           True),
                          ([[2, 'Accepted', 3], [2, 'Rejected', 3], [2, 'Accepted', 3]], None,
                           True)])
def test_minimizer_has_converged(mock_history, min_steps, expected):
    """
    Tests that the has_converged method returns the expected boolean for a number of mocked minimizer histories.
    """
    parameter = Parameters(Parameter(name='A', value=None))
    for minimizer_name in MinimizerFactory.get_minimizer_names():
        minim = MinimizerFactory.create_minimizer(minimizer_name, parameter=parameter)
        minim._history = mock_history
        if min_steps:
            assert minim.has_converged(min_steps=min_steps) == expected
        else:
            assert minim.has_converged() == expected


def test_minimizer_fixed_parameter():
    """
    Test that a ``ValueError`` is raised when passing a fixed ``Parameter``
    """

    parameters = [Parameter(name='fixed', value=1., fixed=True)]
    with pytest.raises(ValueError):
        for minimizer_name in MinimizerFactory.get_minimizer_names():
            minim = MinimizerFactory.create_minimizer(minimizer_name, parameters)


def test_minimizer_tied_parameter():
    """
    Test that a ``ValueError`` is raised when passing a tied ``Parameter``
    """

    target_parameter = Parameter(name='target', value=1., )
    tied_parameter = Parameter(name='tied', value=1., )
    tied_parameter.set_tie(target_parameter, ' * 2')
    parameters = [tied_parameter]
    with pytest.raises(ValueError):
        for minimizer_name in MinimizerFactory.get_minimizer_names():
           minim = MinimizerFactory.create_minimizer(minimizer_name, parameters)

