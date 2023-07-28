"""
Tests the GPO minimizer class
"""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from MDMC.refinement.minimizers.GPO import GPO
from MDMC.refinement.minimizers.minimizer_factory import MinimizerFactory
from MDMC.MD.parameters import Parameters, Parameter


class MockControl:

    def __init__(self, n_steps: int):
        self.n_steps = n_steps


@pytest.fixture(scope="module")
def mockcontrol():

    def _mockcontrol(n_steps: int = 4) -> MockControl:
        return MockControl(n_steps=n_steps)

    return _mockcontrol


@pytest.fixture
def parameters():
    """
    A fixture returning two arbitrary `Parameter` objects
    wrapped in a `Parameters` collection.
    """
    return Parameters([Parameter(name='parameter1', value=1.),
                Parameter(name='parameter2', value=2.)])


@pytest.fixture
def constrained_parameters():
    """
    A fixture returning two arbitrary `Parameter` objects with constraints on their values,
    wrapped in a `Parameters` collection.
    """
    return Parameters([Parameter(name='parameter1', value=1., constraints=(0.5, 2.0)),
                Parameter(name='parameter2', value=2., constraints=(1.0, 4.0))])


@pytest.fixture
def GPO_with_history(mockcontrol, parameters):
    """
    Creates an instance of GPO with a 10-step history

    Returns
    -------
    GPO
        A GPO object with a history of 10 steps
    """

    minimizer = GPO(mockcontrol(10), parameters)
    for i in range(1, 11):
        minimizer.step(FoM=i)
    return minimizer


@pytest.fixture
def mocked_df():
   return pd.DataFrame(
        columns=["Unnamed: 0", "FoM", "parameter1 (#7)", "parameter2 (#8)"],
        data=[
            [0, 1, 1.0, 2.0],
            [1, 2, 1.0263066427512766, 2.2784431236642697],
            [2, 3, 1.0563332898940743, 1.5261781662556804],
            [3, 4, 0.9517098265051485, 2.578890522713669],
            [4, 5, 1.2970476059280804, 2.203879231558817],
            [5, 6, 0.7892038323388955, 1.491195941884538],
            [6, 7, 0.93540608596101, 1.8776663534533826],
            [7, 8, 0.855686055831339, 2.4710408940692625],
            [8, 9, 0.7105919182646769, 1.9649678706679081],
            [9, 10, 1.1302665513264398, 1.4146366407329378]
        ])


@pytest.fixture
def correct_output_data():
    return [(1.0, 2.0), 1.0, (1.0, 2.0), 1.0]


@pytest.mark.parametrize('mock_history, expected',
                         [([[3, 4], [2, 3], [2, 3]], False),
                          ([[3, 4], [2, 3],
                            [2, 3], [2, 3]], True)])
def test_GPO_has_converged(mockcontrol, mock_history, expected):
    """Test that the array of points to be simulated is created correctly"""
    parameter = Parameters(Parameter(name='A', value=1))
    gpo = MinimizerFactory.create_minimizer('GPO', mockcontrol(n_steps=4), parameter)
    gpo._history = mock_history
    assert gpo.has_converged() == expected


def test_GPO_step(mockcontrol):
    """Tests GPO is able to find the minima of a single cycle of a cosine function"""
    parameter = Parameters(Parameter(name='a', value=1.5, constraints=[-2., 4.]))
    gpo = MinimizerFactory.create_minimizer('GPO', mockcontrol(n_steps=100), parameter, n_points=100)
    gpo._history=[]
    for _ in range(25):
        x = parameter['a'].value
        FoM=np.cos(x)+3.0
        gpo.step(FoM)
    assert np.allclose([gpo.predicted_min_pos], [np.pi], atol=1e-2)


def test_GPO_set_parameter_values(mockcontrol, constrained_parameters):
    """Tests set_parameter_values can set values correctly"""

    gpo = MinimizerFactory.create_minimizer('GPO', mockcontrol(), constrained_parameters,
                                            n_points=3)
    gpo.set_parameter_values(['parameter1'], [1.9])
    assert gpo.parameters['parameter1'].value == 1.9

    gpo.set_parameter_values(['parameter1', 'parameter2'], [0.6, 1.56])
    assert gpo.parameters['parameter1'].value == 0.6
    assert gpo.parameters['parameter2'].value == 1.56

    with pytest.raises(ValueError):
        gpo.set_parameter_values(['parameter1'], [0.0])
    with pytest.raises(ValueError):
        gpo.set_parameter_values(['parameter2'], [7.0])


@pytest.mark.parametrize("has_converged_value",
                         [True, False])
def test_converge_message_in_output(GPO_with_history, has_converged_value):
    """Tests that the convergence message is present in the final output"""

    with patch("MDMC.refinement.minimizers.GPO.GPO.has_converged",
               autospec=True,
               return_value=has_converged_value):

        converged = GPO_with_history.has_converged()
        output_message = GPO_with_history.present_result()
        if converged:
            assert "The refinement has finished" in output_message
        else:
            assert "The refinement has not finished" in output_message


def test_GPO_FoM_and_coordinates_in_output(GPO_with_history, correct_output_data, mocked_df):
    """Tests that the correct values are present in the final output"""

    with patch("MDMC.refinement.minimizers.GPR.pd.read_csv", autospec=True, return_value=mocked_df):
        output_data = GPO_with_history.extract_result()
        output_string = GPO_with_history.format_result_string(output_data)

        assert str(correct_output_data[0]) in output_string
        assert str(correct_output_data[2]) in output_string

        assert np.allclose(correct_output_data[1], output_data[1], atol=0.0001, equal_nan=False)
        assert np.allclose(correct_output_data[3], output_data[3], atol=0.0001, equal_nan=False)
