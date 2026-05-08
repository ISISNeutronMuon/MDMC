"""
Tests the Covariance Matrix Adaptation Evolution Strategy minimizer.
"""
import random
from unittest.mock import PropertyMock, patch

import numpy as np
import pandas
import pytest

from MDMC.MD import Parameter, Parameters
from MDMC.refinement import minimizers
from MDMC.refinement.minimizers.CMAES import CMAES
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
def CMAES_with_history(parameters):
    """
    Creates an instance of CMAES with a random, 10-step history

    Returns
    -------
        A CMAES object with a random history of 10 steps
    """
    minimizer = CMAES(parameters)
    randomizer = random.Random()
    for _ in range(60):
        minimizer.step(FoM=randomizer.uniform(0.1, 1000))
    return minimizer


@pytest.mark.parametrize("has_converged_value",
                         [True, False])
def test_converge_message_in_output_string(CMAES_with_history, has_converged_value):
    """Tests that the convergence message is correct in the final output"""
    with patch("MDMC.refinement.minimizers.CMAES.CMAES.has_converged",
               autospec=True,
               return_value=has_converged_value):
        converged = CMAES_with_history.has_converged()
        output_message = CMAES_with_history.present_result()
        if converged:
            assert "The refinement has converged" in output_message
        else:
            assert "The refinement has not converged" in output_message
