"""
Tests the Metropolis-Hastings minimizer
"""
import numpy as np
import pytest

from MDMC.MD import Parameter
from MDMC.refinement import minimizers
from MDMC.refinement.minimizers.minimizer_factory import MinimizerFactory
from tests.refinement.test_minimizer import mock_change_parameters


@pytest.fixture
def parameters():
    """
    Returns
    -------
    list
        A `list` of ``Parameter`` objects with a variety of `name` and
        `value` attributes
    """

    return ([Parameter(name='A', value=1.),
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
    original_values = [p.value for p in parameters]
    changed_values = [p.value * 2 for p in parameters]
    mmc = MinimizerFactory.create_minimizer('MMC', parameters)

    # Monkeypatch both the state change and the parameter change
    monkeypatch.setattr(minimizers.MMC.MMC, 'change_state', mock_change_state)
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
