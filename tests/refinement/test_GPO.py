"""
Tests the GPO minimizer class
"""

from unittest.mock import patch, ANY

import numpy as np
import pandas as pd
import pytest

from skopt import Optimizer

from MDMC.refinement.minimizers.minimizer_abs import Minimizer
from MDMC.refinement.minimizers.minimizer_factory import MinimizerFactory
from MDMC.refinement.minimizers.GPR import GPR
from MDMC.refinement.minimizers.GPO import GPO

from MDMC.MD.parameters import Parameters, Parameter


@pytest.mark.parametrize('mock_history, min_steps, expected',
                         [([[3, 'Accepted', 1, 1, 4], [2, 'Accepted', 1, 1, 3], [2, 'Accepted', 1, 1, 3]], 4, False),
                          ([[3, 'Accepted', 1, 1, 4], [2, 'Accepted', 1, 1, 3], [2, 'Accepted', 1, 1, 3]], 3, True),
                          ([[2, 'Accepted', 1, 1, 3], [2, 'Accepted', 1, 1, 3], [2, 'Accepted', 1, 1, 3]], 2, True)])
def test_GPO_has_converged(mock_history, min_steps, expected):
    """Test that the array of points to be simulated is created correctly"""
    parameter = Parameters(Parameter(name='A', value=1))
    gpo = MinimizerFactory.create_minimizer('GPO', parameter, n_points=min_steps)
    gpo._history = mock_history
    assert gpo.has_converged() == expected


def test_GPO_step():
    """Tests GPO is able to find the minima of a single cycle of a sine function and then return the result"""
    def cosine_function(x: float) -> float:
        return np.cos(x)

    parameter = Parameters(Parameter(name='sine', value=1.5, constraints=[-2., 4.]))
    gpo = MinimizerFactory.create_minimizer('GPO', parameter, n_points=100)
    gpo._history=[]
    for i in range(30):
        x = parameter['sine'].value
        FoM=np.cos(x)+3.0
        gpo.step(FoM)
    print(x)
