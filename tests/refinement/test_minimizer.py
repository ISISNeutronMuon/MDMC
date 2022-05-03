"""Tests the Minimizer class and subclasses
"""

from tempfile import NamedTemporaryFile

from unittest.mock import patch

import numpy as np
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
    list
        A `list` of ``Parameter`` objects with a variety of `name` and
        `value` attributes
    """

    # note these parameters are listed in alphabetical order
    # as that is the order in which they are sorted when added to a Minimizer object
    return([Parameter(name='A', value=1.),
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

    # Ignore pylint error as abstract class is mocked
    # pylint: disable=abstract-class-instantiated
    minim = Minimizer(parameters)
    assert np.all(minim.parameters == np.array(parameters))


@patch.multiple(Minimizer, __abstractmethods__=set())
def test_minimizer_init_invalid_parameters(parameters):

    """
    Test initializing ``Minimizer`` with fixed parameters, which should raise a
    `ValueError`
    """

    # Ignore pylint error as abstract class is mocked
    # pylint: disable=abstract-class-instantiated
    fixed_parameter = Parameter(name='charge', value=1.)
    fixed_parameter.fixed = True

    with pytest.raises(ValueError):
        Minimizer(parameters + [fixed_parameter])


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
    for minimizer_name in MinimizerFactory.get_minimizer_names():

        minim = MinimizerFactory.create_minimizer(minimizer_name, parameters[slice(*p_slice)])
        assert minim.history_columns == ['FoM', 'Change state'] + columns


def mock_change_parameters(self, parameters):

    """
    Mock of minimizer.change_parameters which doubles each ``Parameter`` value
    """

    for p in parameters:
        p.value *= 2


def test_mmc_step_accepted(monkeypatch, parameters):

    """
    Tests that the ``MMC`` minimizer increments with the correct step when the
    state change is accepted (i.e. ``MMC.change_state`` returns `True`)

    This includes testing that:

        - the old FoM is set to whatever the FoM provided is
        - the old parameter values are changed to the current parameter values
        - the state changed attribute is True
        - the history is correctly updated
        - the parameters are correctly changed (using the current parameters)
    """

    def mock_change_state(self):

        return True

    # The original parameter values should be added to the history, and the
    # changed values should be 2x these (as determined by
    # mock_change_parameters)
    original_values = [p.value for p in parameters]
    changed_values = [p.value * 2 for p in parameters]
    mmc = MinimizerFactory.create_minimizer('MMC', parameters)

    # Monkeypatch both the state change and the parameter change
    monkeypatch.setattr( minimizers.MMC.MMC, 'change_state', mock_change_state)
    monkeypatch.setattr(minimizers.MMC.MMC, 'change_parameters',
                        mock_change_parameters)

    FoM = 1000.
    mmc.step(FoM)

    assert mmc.FoM_old == FoM
    assert np.all(mmc.parameters_old_values == np.array(original_values))
    assert mmc.state_changed is True
    assert [p.value for p in mmc.parameters] == changed_values
    assert mmc._history == [[FoM, 'Accepted'] + original_values]


def test_mmc_step_rejected(monkeypatch, parameters):

    """
    Tests that the ``MMC`` minimizer increments with the correct step when the
    state change is rejected (i.e. ``MMC.change_state`` returns `False`)

    This includes testing that:

        - the current FoM is set to the old FoM
        - the current parameters are reset to their old values
        - the state changed attribute is False
        - the history is correctly updated
        - the parameters are correctly changed (using the old parameters)
    """

    def mock_change_state(self):

        return False

    # The original parameter values should be added to the history, and the
    # changed values should be 2x the old values which the MMC already
    # possesses.  As these are not set when MMC is initialised, set these
    # manually to something arbitrary.
    mmc = MinimizerFactory.create_minimizer('MMC', parameters)
    mmc.parameters_old_values = np.arange(len(parameters))
    original_FoM = mmc.FoM_old
    original_values = [p.value for p in parameters]
    expected_values = list(mmc.parameters_old_values * 2)

    # Monkeypatch both the state change and the parameter change
    monkeypatch.setattr(minimizers.MMC.MMC, 'change_state', mock_change_state)
    monkeypatch.setattr(minimizers.MMC.MMC, 'change_parameters',
                        mock_change_parameters)

    FoM = 1000.
    mmc.step(FoM)

    assert mmc.FoM == original_FoM
    assert mmc.state_changed is False
    assert np.all(mmc.parameters_old_values == np.arange(len(parameters)))
    assert [p.value for p in mmc.parameters] == expected_values
    assert mmc._history == [[FoM, 'Rejected'] + original_values]


def test_minimizer_change_parameters(parameters):

    """
    Tests that the parameters change by the expected amount when given a mocked
    distribution which always returns 1.
    """

    def mock_distribution(low: float, high: float, size: int):
        return np.ones(size)

    expected_values = [2 * p.value for p in parameters]
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

    parameters = [Parameter(name='constraints', value=1., constraints=(0.5, 1.5)),
                  Parameter(name='constraints', value=1., constraints=(0.5, 1.5))]

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
    for minimizer_name in MinimizerFactory.get_minimizer_names():
        minim = MinimizerFactory.create_minimizer(minimizer_name, parameters, MC_norm=MC_norm)
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

    target_parameter = Parameter(name='target', value=1.,)
    tied_parameter = Parameter(name='tied', value=1.,)
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
    name, points = gpr.create_parameter_point_array(parameters)
    assert np.allclose(points[0], (0.46, 1.1), rtol=1e-5)
    assert np.allclose(points[1], (0.46, 1.9), rtol=1e-5)
    assert np.allclose(points[2], (0.94, 1.1), rtol=1e-5)
    assert np.allclose(points[3], (0.94, 1.9), rtol=1e-5)

    constrained_parameters = Parameters([Parameter(name='parameter1', value=1., constraints=(0.5,2.0)), 
                            Parameter(name='parameter2', value=2.,  constraints=(1.0,4.0))])
    _, points = gpr.create_parameter_point_array(constrained_parameters)
    assert np.allclose(points[0], [0.5, 1.0], rtol=1e-5)
    assert np.allclose(points[2], [2.0, 1.0], rtol=1e-5)
    
    gpr = MinimizerFactory.create_minimizer('GPR', constrained_parameters, n_points=4, hypercube=True)
    names, points = gpr.create_parameter_point_array(constrained_parameters, points=4)
    assert len(points) == 4
    assert np.logical_and(np.array(points)[:,0]>=constrained_parameters.filter_name(str(names[0]))[0].constraints[0], np.array(points)[:,0]<=constrained_parameters.filter_name(str(names[0]))[0].constraints[1]).all()
    assert np.logical_and(np.array(points)[:,1]>=constrained_parameters.filter_name(str(names[1]))[0].constraints[0], np.array(points)[:,1]<=constrained_parameters.filter_name(str(names[1]))[0].constraints[1]).all()


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
def test_global_minimum_position(FoMs, coordinates, expected):
    constrained_parameter = Parameters([Parameter(name='parameter1', value=1., constraints=(0.5,2.0))])
    gpr = MinimizerFactory.create_minimizer('GPR', constrained_parameter, n_points=3)
    min_coord, min_FoM = gpr.global_minimum_position(FoMs, coordinates)
    assert np.allclose([min_coord, min_FoM] == expected, rtol=1e-5)

@pytest.mark.parametrize('points,FoMs,expected',
[([[1],[2],[3]], [1,2,3], [1,1])])
def test_present_results(points,FoMs,expected):
    with patch("MDMC.refinement.minimizers.GPR.GPR.GPR_fit", autospec=True, return_value=None):
        with patch("MDMC.refinement.minimizers.GPR.GPR.GPR_predict", autospec=True, return_value=(points, FoMs)):
            constrained_parameter = Parameters([Parameter(name='parameter1', value=1., constraints=(0.5,2.0))])
            gpr = MinimizerFactory.create_minimizer('GPR', constrained_parameter, n_points=3)
            coord, FoM = gpr.present_result()
            assert [coord, FoM] == expected
    