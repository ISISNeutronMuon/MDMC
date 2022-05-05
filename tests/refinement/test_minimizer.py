"""
Tests the Minimizer base class
"""

from tempfile import NamedTemporaryFile

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

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
    Tests that the history columns for the ``MMC`` minimizer are as expected,
    including the names of the ``Parameter`` objects which are refined
    """
    parameter_slice = Parameters(list(parameters.values())[slice(*p_slice)])

    for minimizer_name in MinimizerFactory.get_minimizer_names():
        minim = MinimizerFactory.create_minimizer(minimizer_name, parameter_slice)
        assert minim.history_columns == ['FoM', 'Change state'] + columns


def mock_change_parameters(self, parameters):
    """
    Mock of minimizer.change_parameters which doubles each ``Parameter`` value
    """

    for p in parameters:
        p.value *= 2


def test_minimizer_change_parameters(parameters):
    """
    Tests that the parameters change by the expected amount when given a mocked
    distribution which always returns 1.
    """

    def mock_distribution(low: float, high: float, size: int):
        return np.ones(size)

    expected_values = [2 * p.value for p in parameters.values()]
    for minimizer_name in MinimizerFactory.get_minimizer_names():
        minim = MinimizerFactory.create_minimizer(minimizer_name, parameters)
        minim.distribution = mock_distribution
        minim.change_parameters(minim.parameters)
        assert [p.value for p in minim.parameters] == expected_values


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
        assert [p.value for p in minim.parameters] == expected_values


@pytest.mark.parametrize('FoM, FoM_old',
                         [(10., 20.),
                          (10., 20.),
                          (10., 20.),
                          (1.e+10, 1.e+10),
                          (0., 0.)])
def test_minimizer_change_state_FoM_le(monkeypatch, parameters, FoM, FoM_old):
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

    minim = MinimizerFactory.create_minimizer('MMC', parameters, MC_norm=MC_norm)
    minim.FoM_old = FoM_old
    minim.FoM = FoM
    monkeypatch.setattr(np.random, 'random', mock_random)
    assert minim.change_state() == change


@pytest.mark.parametrize('mock_history, min_steps, expected',
    [([[3,'Accepted',4],[2,'Accepted',3],[1,'Accepted',2]], None, False),
    ([[3,'Accepted',4],[2,'Accepted',3],[2,'Accepted',2]], None, False),
    ([[3,'Accepted',4],[2,'Rejected',3],[2,'Accepted',3]], None, False),
    ([[3,'Accepted',4],[2,'Accepted',3],[1,'Accepted',3]], None, False),
    ([[2,'Accepted',4],[2,'Rejected',4],[2,'Rejected',4]], None, False),
    ([[3,'Accepted',4],[2,'Accepted',3],[2,'Accepted',3]],4, False),
    ([[3,'Accepted',4],[2,'Accepted',3],[2,'Accepted',3]], None, True),
    ([[2,'Accepted',3],[2,'Rejected',3],[2,'Accepted',3]], None, True)])
def test_MMC_minimizer_has_converged(mock_history, min_steps, expected):
    """
    Tests that the has_converged method returns the expected boolean for a number of mocked minimizer histories.
    """
    parameter = [Parameter(name='A', value=1.)]
    minim = MinimizerFactory.create_minimizer('MMC', parameter=parameter)
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


def test_GPR_parameter_point_array():
    """
    Test that the array of points to be simulated is created correctly
    """
    parameters = Parameters([Parameter(name='parameter1', value=1.), 
                    Parameter(name='parameter2', value=2.)])
    gpr = MinimizerFactory.create_minimizer('GPR', parameters, n_points=2)
    points = gpr.parameter_point_array
    assert np.allclose(points[0], (0.8, 1.6), rtol=1e-5)
    assert np.allclose(points[1], (0.8, 2.4), rtol=1e-5)
    assert np.allclose(points[2], (1.2, 1.6), rtol=1e-5)
    assert np.allclose(points[3], (1.2, 2.4), rtol=1e-5)

    constrained_parameters = Parameters([Parameter(name='parameter1', value=1., constraints=(0.5,2.0)), 
                            Parameter(name='parameter2', value=2.,  constraints=(1.0,4.0))])
    _, points = gpr.create_parameter_point_array(constrained_parameters)
    assert np.allclose(points[0], [0.5, 1.0], rtol=1e-5)
    assert np.allclose(points[2], [2.0, 1.0], rtol=1e-5)
    
    gpr = MinimizerFactory.create_minimizer('GPR', constrained_parameters, n_points=4, hypercube=True)
    points = gpr.parameter_point_array
    par1_constraints = constrained_parameters['parameter1'].constraints
    par2_constraints = constrained_parameters['parameter2'].constraints

    assert len(points) == 4
    assert np.logical_and(np.array(points)[:,0]>=par1_constraints[0], np.array(points)[:,0]<=par1_constraints[1]).all()
    assert np.logical_and(np.array(points)[:,1]>=par2_constraints[0], np.array(points)[:,1]<=par2_constraints[1]).all()


def test_GPR_create_bounds():
    constrained_parameter = Parameters([Parameter(name='parameter1', value=1., constraints=(0.5,2.0))])
    gpr = MinimizerFactory.create_minimizer('GPR', constrained_parameter, n_points=3)
    lower_bound, upper_bound = gpr.create_bounds(constrained_parameter[0])
    assert np.allclose([lower_bound, upper_bound], [constrained_parameter[0].constraints[0], constrained_parameter[0].constraints[1]], rtol=1e-5)

@pytest.mark.parametrize('FoMs,coordinates,expected',
    [([2, 3, 0, 1, 4], 
    [[0,0], [0,1], [1,0], [1,1], [2,0]], 
    [[1,0], 0]),
    ([2], 
    [[0,0,1]], 
    [[0,0,1], 2]),
    ([0.01, 0.020, 0.01, 6], 
    [[0.1,0.1,0.1],[0.1,0.1,1],[0.1,1,1],[1,1,1]], 
    [[0.1,0.1,0.1], 0.01])])
def test_GPR_global_minimum_position(FoMs, coordinates, expected):
    constrained_parameter = Parameters([Parameter(name='parameter1', value=1., constraints=(0.5,2.0))])
    gpr = MinimizerFactory.create_minimizer('GPR', constrained_parameter, n_points=3)
    min_coord, min_FoM = gpr.global_minimum_position(FoMs, coordinates)
    assert np.allclose(min_coord, expected[0], rtol=1e-5)
    assert np.allclose(min_FoM, expected[1], rtol=1e-5)

@pytest.mark.parametrize('points,FoMs,expected',
[([[1],[2],[3]], [1,2,3], [[1],1]),
([[1,0],[2,0],[3,0],[4,0]], [0.1,2,3,0], [[4,0],0])])
def test_GPR_present_results(points,FoMs,expected):
    with patch("MDMC.refinement.minimizers.GPR.GPR.GPR_fit", autospec=True, return_value=None):
        with patch("MDMC.refinement.minimizers.GPR.GPR.GPR_predict", autospec=True, return_value=(points, FoMs)):
            gpr = MinimizerFactory.create_minimizer('GPR', Parameters(), n_points=3)
            coord, FoM = gpr.present_result()
            assert np.allclose(coord, expected[0], rtol=1e-5)
            assert np.allclose(FoM, expected[1], rtol=1e-5)

def test_GPR_create_bounds():
    constrained_parameter = Parameter(name='parameter1', value=1., constraints=(0.5,2.0))
    unconstrained_parameter = Parameter(name='parameter1', value=1.)
    unconstrained_parameter_zero = Parameter(name='parameter1', value=0.0)

    gpr = MinimizerFactory.create_minimizer('GPR', Parameters(), n_points=3)
    lower_bound, upper_bound = gpr.create_bounds(constrained_parameter)
    assert np.allclose([lower_bound, upper_bound], [0.5,2.0], rtol=1e-5)

    lower_bound, upper_bound = gpr.create_bounds(unconstrained_parameter)
    assert np.allclose([lower_bound, upper_bound], [0.8,1.2], rtol=1e-5)

    with pytest.raises(ValueError):
        lower_bound, upper_bound = gpr.create_bounds(unconstrained_parameter_zero)

def test_GPR_set_parameter_values():
    constrained_parameter = Parameters([Parameter(name='parameter1', value=1., constraints=(0.5,2.0)),
                                        Parameter(name='parameter2', value=2., constraints=(0.3,6.0))])
    gpr = MinimizerFactory.create_minimizer('GPR', constrained_parameter, n_points=3)
    gpr.set_parameter_values(['parameter1'], [1.9])
    assert gpr.parameters['parameter1'].value == 1.9

    gpr.set_parameter_values(['parameter1', 'parameter2'], [0.6, 1.56])
    assert gpr.parameters['parameter1'].value == 0.6
    assert gpr.parameters['parameter2'].value == 1.56

    with pytest.raises(ValueError):
        gpr.set_parameter_values(['parameter1'], [0.0])
    with pytest.raises(ValueError):
        gpr.set_parameter_values(['parameter2'], [7.0])


def test_GPR_fit():
    mocked_df = pd.DataFrame(data=[
                                    [0,356.0792119015762,'Accepted',0.2,2.6],
                                    [1,2306.5433713234515,'Accepted',1.8,2.6]],
                                    columns=['','FoM','Change state','epsilon','sigma'])
    with patch("MDMC.refinement.minimizers.GPR.GPR.pd.read_csv", autospec=True, return_value=mocked_df):
        pass