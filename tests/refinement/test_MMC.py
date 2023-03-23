"""
Tests the Metropolis-Hastings minimizer
"""
import random
from unittest.mock import patch, PropertyMock

import numpy as np
import pandas
import pytest

from MDMC.MD import Parameter, Parameters
from MDMC.refinement import minimizers
from MDMC.refinement.minimizers.MMC import MMC
from MDMC.refinement.minimizers.minimizer_factory import MinimizerFactory


class MockControl:

    def __init__(self):
        pass


@pytest.fixture(scope="module")
def mockcontrol():

    _mockcontrol = MockControl()
    return _mockcontrol


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


@pytest.fixture
def MMC_with_history(mockcontrol, parameters):
    """
    Creates an instance of MMC with a random, 10-step history

    Returns
    -------
        A MMC object with a random history of 10 steps
    """
    minimizer = MMC(mockcontrol, parameters)
    randomizer = random.Random()
    for i in range(10):
        minimizer.step(FoM=randomizer.uniform(0.1, 1000))
    return minimizer


def mock_change_parameters(self):
    """
    Mock of minimizer.change_parameters which doubles each ``Parameter`` value
    """

    for p in self.parameters:
        self.parameters[p].value *= 2


def test_mmc_step_accepted(monkeypatch, mockcontrol, parameters):
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
    mmc = MinimizerFactory.create_minimizer('MMC', mockcontrol, parameters)

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


def test_mmc_step_rejected(monkeypatch, mockcontrol, parameters):
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
    mmc = MinimizerFactory.create_minimizer('MMC', mockcontrol, parameters)
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
def test_MMC_has_converged(mockcontrol, mock_history, min_steps, expected):
    """
    Tests that the has_converged method returns the expected boolean for a number of
    mocked minimizer histories.
    """
    parameter = Parameters(Parameter(name='A', value=None))
    if min_steps:
        minim = MinimizerFactory.create_minimizer('MMC', mockcontrol, parameter, min_steps=min_steps)
    else:
        minim = MinimizerFactory.create_minimizer('MMC',  mockcontrol, parameter)
    minim._history = mock_history
    assert minim.has_converged() == expected


def test_MMC_change_parameter(mockcontrol, parameters):
    """
    Tests that unconstrained parameters change by the expected amount when given a mocked
    distribution which always returns 1 and constrained parameters do not exceed their
    max/min values.
    """

    def mock_distribution(low: float, high: float, size: int):
        return np.ones(size)

    minim = MinimizerFactory.create_minimizer('MMC', mockcontrol, parameters)
    expected_values = {p: 2 * parameters[p].value for p in parameters}
    minim.distribution = mock_distribution
    minim.change_parameters()
    for p in minim.parameters:
        assert minim.parameters[p].value == expected_values[p]

    def mock_distribution2(low: float, high: float, size: int):
        return np.array([1., -1.])

    parameters = Parameters([Parameter(name='constraints', value=1., constraints=(0.5, 1.5)),
                             Parameter(name='constraints_2', value=1., constraints=(0.5, 1.5))])
    # Expect values to be set to the upper/lower limit
    expected_values = [1.5, 0.5]
    minim = MinimizerFactory.create_minimizer('MMC', mockcontrol, parameters)
    minim.distribution = mock_distribution2
    minim.change_parameters()
    assert [p.value for p in minim.parameters.values()] == expected_values


@pytest.mark.parametrize('FoM, FoM_old, MC_norm, change',
                         [(21., 20., 1., False),
                          (21., 20., 2., True),
                          (21., 20., 100., True),
                          (40., 20., 29., True),
                          (40., 20., 28., False),
                          (60., 40., 29., True),
                          (60., 40., 28., False)])
def test_MMC_change_state_FoM_gt(monkeypatch, mockcontrol, parameters, FoM, FoM_old,
                                 MC_norm, change):
    """
    Tests that the state changes correctly given an FoM, old FoM, and MC norm,
    where the FoM is greater than the old FoM, and the return of
    ``np.random.random`` is 0.5
    """

    def mock_random():
        return 0.5

    minim = MinimizerFactory.create_minimizer('MMC', mockcontrol, parameters, MC_norm=MC_norm)
    minim.FoM_old = FoM_old
    minim.FoM = FoM
    monkeypatch.setattr(np.random, 'random', mock_random)
    assert minim.change_state() == change


@pytest.mark.parametrize('FoM, FoM_old',
                         [(10., 20.),
                          (10., 20.),
                          (10., 20.),
                          (1.e+10, 1.e+10),
                          (0., 0.)])
def test_MMC_change_state_FoM_le(monkeypatch, mockcontrol, parameters, FoM, FoM_old):
    """
    Tests that the state always changes (i.e. returns True) given an FoM, old
    FoM, and MC norm, where the FoM is less than or equal to old FoM, and the
    return of ``np.random.random`` is 0.999
    """

    def mock_random():
        return 0.999999

    minim = MinimizerFactory.create_minimizer('MMC', mockcontrol, parameters, MC_norm=1.0)
    minim.FoM_old = FoM_old
    minim.FoM = FoM
    monkeypatch.setattr(np.random, 'random', mock_random)
    assert minim.change_state() is True


@pytest.mark.parametrize("has_converged_value",
                         [True, False])
def test_converge_message_in_output_string(MMC_with_history, has_converged_value):
    """Tests that the convergence message is correct in the final output"""
    with patch("MDMC.refinement.minimizers.MMC.MMC.has_converged",
               autospec=True,
               return_value=has_converged_value):
        converged = MMC_with_history.has_converged()
        output_message = MMC_with_history.present_result()
        if converged:
            assert "The refinement has converged" in output_message
        else:
            assert "The refinement has not converged" in output_message


@pytest.mark.parametrize('mock_history, FoMs, expected',
                         [(pandas.DataFrame(data=[
                             [123.4, "Accepted", 23.453, 8.],
                             [235.6, "Rejected", 23.567, 7.85],
                             [100.2, "Accepted", 24.658, 6.5]
                         ],
                             columns=["FoM", "Change state", "A (#1)", "B (#2)"]),
                           (100.2, 100.2),
                           ((24.658, 6.5), (24.658, 6.5))),
                             (pandas.DataFrame(data=[
                                 [123.4, "Accepted", 22.453, 8.],
                                 [34.6, "Accepted", 23.567, 7.85],
                                 [45.2, "Rejected", 20.655, 5.5]
                             ], columns=["FoM", "Change state", "A (#1)", "B (#2)"]),
                              (34.6, 45.2),
                              ((23.567, 7.85), (20.655, 5.5))
                             ),
                             (pandas.DataFrame(data=[
                                 [123.4, "Accepted", 23.453, 8.],
                                 [235.6, "Rejected", 23.567, 7.85],
                                 [145.2, "Rejected", 24.658, 6.5]
                             ], columns=["FoM", "Change state", "A (#1)", "B (#2)"]),
                              (123.4, 145.2),
                              ((23.453, 8.), (24.658, 6.5))
                             )])
class TestParametrized:
    """A class of tests that shares parametrized data"""

    def test_MMC_extract_result(self, mock_history, mockcontrol, FoMs, expected):
        """Tests that the correct values are extracted from the history"""
        params = Parameters()
        with patch("MDMC.refinement.minimizers.MMC.MMC.history", new_callable=PropertyMock) as hist:
            hist.return_value = mock_history
            with patch("MDMC.refinement.minimizers.MMC.MMC.history_columns",
                       new_callable=PropertyMock) as columns:
                columns.return_value = list(mock_history.columns)
                mmc = MinimizerFactory().create_minimizer("MMC", mockcontrol, params)
                output_data = mmc.extract_result()
                assert FoMs[0] in output_data
                assert FoMs[1] in output_data
                assert expected[0] == output_data[2]
                assert expected[1] == output_data[0]

    def test_MMC_FoM_and_coordinates_in_output(self, mock_history, mockcontrol, FoMs, expected):
        """Tests that the correct coordinates are present in the final output"""
        params = Parameters()
        with patch("MDMC.refinement.minimizers.MMC.MMC.history", new_callable=PropertyMock) as hist:
            hist.return_value = mock_history
            with patch("MDMC.refinement.minimizers.MMC.MMC.history_columns",
                       new_callable=PropertyMock) as columns:
                columns.return_value = list(mock_history.columns)
                mmc = MinimizerFactory().create_minimizer("MMC", mockcontrol, params)
                output_string = mmc.present_result()
                assert str(expected[0]) in output_string
                assert str(expected[1]) in output_string
                assert str(FoMs[0]) in output_string
                assert str(FoMs[1]) in output_string