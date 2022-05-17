"""
Tests the Metropolis-Hastings minimizer
"""
import numpy as np
import pytest

from unittest.mock import patch

from MDMC.MD import Parameter, Parameters
from MDMC.refinement import minimizers
from MDMC.refinement.minimizers.minimizer_factory import MinimizerFactory
from tests.refinement.test_minimizer import mock_change_parameters
import numpy as np
import pytest


@pytest.fixture
def parameters():
    """
    Returns
    -------
    list
        A `list` of ``Parameter`` objects with a variety of `name` and
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
    original_values = [parameters[p].value for p in parameters]
    changed_values = [parameters[p].value * 2 for p in parameters]
    mmc = MinimizerFactory.create_minimizer('MMC', parameters)

    # Monkeypatch both the state change and the parameter change
    monkeypatch.setattr(minimizers.MMC.MMC, 'change_state', mock_change_state)
    monkeypatch.setattr(minimizers.MMC.MMC, 'change_parameters',
                        mock_change_parameters)

    FoM = 1000.
    mmc.step(FoM)

    assert mmc.FoM_old == FoM
    assert np.all(np.array(list(mmc.parameters_old_values.values())) == np.array(original_values))
    assert mmc.state_changed is True
    assert [mmc.parameters[p].value for p in mmc.parameters] == changed_values
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
    mmc.parameters_old_values = {p: v for p, v in zip(parameters, np.arange(len(parameters)))}
    original_FoM = mmc.FoM_old
    original_values = [parameters[p].value for p in parameters]
    expected_values = list(np.arange(len(parameters)) * 2)

    # Monkeypatch both the state change and the parameter change
    monkeypatch.setattr(minimizers.MMC.MMC, 'change_state', mock_change_state)
    monkeypatch.setattr(minimizers.MMC.MMC, 'change_parameters',
                        mock_change_parameters)

    FoM = 1000.
    mmc.step(FoM)

    assert mmc.FoM == original_FoM
    assert mmc.state_changed is False
    assert np.all(np.array(list(mmc.parameters_old_values.values())) == np.arange(len(parameters)))
    assert [parameters[p].value for p in mmc.parameters] == expected_values
    assert mmc._history == [[FoM, 'Rejected'] + original_values]


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
    minim = MinimizerFactory.create_minimizer('MMC', parameters)
    minim.distribution = mock_distribution
    minim.change_parameters(minim.parameters)
    assert [parameters[p].value for p in minim.parameters] == expected_values


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
def test_MMC_minimizer_has_converged(mock_history, min_steps, expected):
    """
    Tests that the has_converged method returns the expected boolean for a number of mocked minimizer histories.
    """
    parameter = Parameters(Parameter(name='A', value=None))
    minim = MinimizerFactory.create_minimizer('MMC', parameter=parameter)
    minim._history = mock_history
    if min_steps:
        assert minim.has_converged(min_steps=min_steps) == expected
    else:
        assert minim.has_converged() == expected

@pytest.mark.parametrize('points,FoMs,expected',
[([[1],[2],[3]], [1,2,3], [[1],1]),
([[1,0],[2,0],[3,0],[4,0]], [0.1,2,3,0], [[4,0],0])])
def test_MMC_present_results(points,FoMs,expected):
    params = Parameters()
    with patch("MDMC.refinement.minimizers.MMC.MMC.history", autospec=True) as hist:
        hist['FoM'].min.return_value = expected[1]
        hist['FoM'].idxmin.return_value = FoMs
        hist.iloc.__getitem__.return_value = points
        mmc = MinimizerFactory.create_minimizer('MMC', params)
        coord = mmc.present_result()
        assert str(expected[0]) in coord
        assert str(expected[1]) in coord


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

def test_MMC_minimizer_change_constrained_parameter():
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
    minim = MinimizerFactory.create_minimizer('MMC', parameters)
    minim.distribution = mock_distribution
    minim.change_parameters(minim.parameters)
    assert [p.value for p in minim.parameters.values()] == expected_values


@pytest.mark.parametrize('FoM, FoM_old, MC_norm, change',
                         [(21., 20., 1., False),
                          (21., 20., 2., True),
                          (21., 20., 100., True),
                          (40., 20., 29., True),
                          (40., 20., 28., False),
                          (60., 40., 29., True),
                          (60., 40., 28., False)])
def test_MMC_minimizer_change_state_FoM_gt(monkeypatch, parameters, FoM, FoM_old,
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