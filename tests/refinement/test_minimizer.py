"""Tests the Minimizer class and subclasses
"""

from tempfile import NamedTemporaryFile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from MDMC.MD.parameters import Parameter
from MDMC.refinement import minimizer


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

    return([Parameter(name='charge', value=1.),
            Parameter(name='charge', value=.5),
            Parameter(name='sigma', value=3.3),
            Parameter(name='epsilon', value=.2),
            Parameter(name='potential_strength', value=1234.),
            Parameter(name='equilibrium_state', value=1.2),
            Parameter(name='A', value=1.),
            Parameter(name='B', value=2.),
            Parameter(name='C', value=3.)])


@patch.multiple('MDMC.refinement.minimizer.Minimizer', __abstractmethods__=set()
               )
def test_minimizer_init(parameters):

    """
    Test initializing ``Minimizer``
    """

    # Ignore pylint error as abstract class is mocked
    # pylint: disable=abstract-class-instantiated
    minim = minimizer.Minimizer(1, parameters)
    assert np.all(minim.parameters == np.array(parameters))


@patch.multiple('MDMC.refinement.minimizer.Minimizer', __abstractmethods__=set()
               )
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
    assert mmc.history_columns == ['FoM', 'Change state'] + columns


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
    mmc = minimizer.MMC(1, parameters)

    # Monkeypatch both the state change and the parameter change
    monkeypatch.setattr(minimizer.MMC, 'change_state', mock_change_state)
    monkeypatch.setattr(minimizer.MMC, 'change_parameters',
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
    mmc = minimizer.MMC(1, parameters)
    mmc.parameters_old_values = np.arange(len(parameters))
    original_FoM = mmc.FoM_old
    original_values = [p.value for p in parameters]
    expected_values = list(mmc.parameters_old_values * 2)

    # Monkeypatch both the state change and the parameter change
    monkeypatch.setattr(minimizer.MMC, 'change_state', mock_change_state)
    monkeypatch.setattr(minimizer.MMC, 'change_parameters',
                        mock_change_parameters)

    FoM = 1000.
    mmc.step(FoM)

    assert mmc.FoM == original_FoM
    assert mmc.state_changed is False
    assert np.all(mmc.parameters_old_values == np.arange(len(parameters)))
    assert [p.value for p in mmc.parameters] == expected_values
    assert mmc._history == [[FoM, 'Rejected'] + original_values]


def test_mmc_change_parameters(parameters):

    """
    Tests that the parameters change by the expected amount when given a mocked
    distribution which always returns 1.
    """

    def mock_distribution(low: float, high: float, size: int):
        return np.ones(size)

    expected_values = [2 * p.value for p in parameters]
    mmc = minimizer.MMC(1, parameters)
    mmc.distribution = mock_distribution
    mmc.change_parameters(mmc.parameters)
    assert [p.value for p in mmc.parameters] == expected_values


def test_mmc_change_constrained_parameter():

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
    mmc = minimizer.MMC(1, parameters)
    mmc.distribution = mock_distribution
    mmc.change_parameters(mmc.parameters)
    assert [p.value for p in mmc.parameters] == expected_values


@pytest.mark.parametrize('FoM, FoM_old, MC_norm',
                         [(10., 20., 1.),
                          (10., 20., 0.0001),
                          (10., 20., 100000.),
                          (1.e+10, 1.e+10, 1.),
                          (0., 0., 100.)])
def test_mmc_change_state_FoM_le(monkeypatch, parameters, FoM, FoM_old,
                                 MC_norm):

    """
    Tests that the state always changes (i.e. returns True) given an FoM, old
    FoM, and MC norm, where the FoM is less than or equal to old FoM, and the
    return of ``np.random.random`` is 0.999
    """

    def mock_random():

        return 0.999999

    mmc = minimizer.MMC(MC_norm, parameters)
    mmc.FoM_old = FoM_old
    mmc.FoM = FoM
    monkeypatch.setattr(np.random, 'random', mock_random)
    assert mmc.change_state() is True


@pytest.mark.parametrize('FoM, FoM_old, MC_norm, change',
                         [(21., 20., 1., False),
                          (21., 20., 2., True),
                          (21., 20., 100., True),
                          (40., 20., 29., True),
                          (40., 20., 28., False),
                          (60., 40., 29., True),
                          (60., 40., 28., False)])
def test_mmc_change_state_FoM_gt(monkeypatch, parameters, FoM, FoM_old,
                                 MC_norm, change):

    """
    Tests that the state changes correctly given an FoM, old FoM, and MC norm,
    where the FoM is greater than the old FoM, and the return of
    ``np.random.random`` is 0.5
    """

    def mock_random():

        return 0.5

    mmc = minimizer.MMC(MC_norm, parameters)
    mmc.FoM_old = FoM_old
    mmc.FoM = FoM
    monkeypatch.setattr(np.random, 'random', mock_random)
    assert mmc.change_state() == change

@pytest.mark.parametrize('mock_history, min_steps, expected',
    [([[3,'Accepted',4],[2,'Accepted',3],[1,'Accepted',2]], None, False),
    ([[3,'Accepted',4],[2,'Accepted',3],[2,'Accepted',2]], None, False),
    ([[3,'Accepted',4],[2,'Rejected',3],[2,'Accepted',3]], None, False),
    ([[3,'Accepted',4],[2,'Accepted',3],[1,'Accepted',3]], None, False),
    ([[2,'Accepted',4],[2,'Rejected',4],[2,'Rejected',4]], None, False),
    ([[3,'Accepted',4],[2,'Accepted',3],[2,'Accepted',3]],4, False),
    ([[3,'Accepted',4],[2,'Accepted',3],[2,'Accepted',3]], None, True),
    ([[2,'Accepted',3],[2,'Rejected',3],[2,'Accepted',3]], None, True)])
def test_minimizer_has_converged(mock_history, min_steps, expected):
    """
    Tests that the has_converged method returns the expected boolean for a number of mocked minimizer histories.
    """
    parameter = [Parameter(name='A', value=None)]
    mmc = minimizer.MMC(MC_norm=1, parameters=parameter)
    mmc._history = mock_history
    if min_steps:
        assert mmc.has_converged(min_steps=min_steps) == expected
    else:
        assert mmc.has_converged() == expected


def test_mmc_fixed_parameter():

    """
    Test that a ``ValueError`` is raised when passing a fixed ``Parameter``
    """

    parameters = [Parameter(name='fixed', value=1., fixed=True)]
    with pytest.raises(ValueError):
        mmc = minimizer.MMC(1, parameters)


def test_mmc_tied_parameter():

    """
    Test that a ``ValueError`` is raised when passing a tied ``Parameter``
    """

    target_parameter = Parameter(name='target', value=1.,)
    tied_parameter = Parameter(name='tied', value=1.,)
    tied_parameter.set_tie(target_parameter, ' * 2')
    parameters = [tied_parameter]
    with pytest.raises(ValueError):
        mmc = minimizer.MMC(1, parameters)
