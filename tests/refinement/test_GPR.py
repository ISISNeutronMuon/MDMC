"""
Tests the GPR minimizer class
"""
from unittest.mock import patch, ANY

import numpy as np
import pandas as pd
import pytest

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from MDMC.MD.parameters import Parameter, Parameters
from MDMC.refinement.minimizers.GPR import GPR
from MDMC.refinement.minimizers.minimizer_factory import MinimizerFactory


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
    Returns multiple constrained parameters

    Returns
    -------
    Parameters
        A collection of constrained parameters
    """
    return Parameters([Parameter(name='parameter1', value=1.),
                Parameter(name='parameter2', value=2.)])


@pytest.fixture
def constrained_parameters():
    """
    Returns multiple constained parameters

    Returns
    -------
    Parameters
        A collection of constrained parameters
    """
    return Parameters([Parameter(name='parameter1', value=1., constraints=(0.5, 2.0)),
                Parameter(name='parameter2', value=2., constraints=(1.0, 4.0))])


@pytest.fixture
def GPR_with_history(mockcontrol, parameters):
    """
    Creates an instance of GPR with a 10-step history

    Returns
    -------
    GPR
        A GPR object with a history of 10 steps
    """

    minimizer = GPR(mockcontrol(10), parameters)
    for i in range(10):
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
    return [(1.0, 2.0), 1.0, np.array([0.99972565, 2.09108368]), 0.9935490487887009]


def test_GPR_parameter_point_array_hypercube(mockcontrol, constrained_parameters):
    """Test that the array of points to be simulated is created correctly"""
    gpr = MinimizerFactory.create('GPR', mockcontrol(), constrained_parameters,
                                            n_points=4)
    points = gpr.parameter_point_array
    par1_constraints = constrained_parameters['parameter1'].constraints
    par2_constraints = constrained_parameters['parameter2'].constraints

    assert len(points) == 4
    assert np.all([np.array(points)[:,0]>=par1_constraints[0],
                   np.array(points)[:,0]<=par1_constraints[1]])

    assert np.all([np.array(points)[:,1]>=par2_constraints[0],
                   np.array(points)[:,1]<=par2_constraints[1]])


def test_GPR_reset_parameters(mockcontrol, parameters):
    """Test that parameters get reset"""
    gpr = MinimizerFactory.create('GPR', mockcontrol(n_steps=2), parameters)

    parameter_values = [p.value for p in gpr.parameters.values()]
    assert np.allclose(parameter_values, (0.85, 1.7), rtol=1e-5)

    gpr.reset_parameters()
    parameter_values = [p.value for p in gpr.parameters.values()]
    assert np.allclose(parameter_values, (1.15, 2.3), rtol=1e-5)

    
@pytest.mark.parametrize('FoMs,coordinates,expected',
    [([2, 3, 0, 1, 4], [[0,0], [0,1], [1,0], [1,1], [2,0]], [[1,0], 0]),
    ([2], [[0,0,1]], [[0,0,1], 2]),
    ([0.01, 0.020, 0.01, 6], [[0.1,0.1,0.1],[0.1,0.1,1],[0.1,1,1],[1,1,1]], [[0.1,0.1,0.1], 0.01])])
def test_GPR_global_minimum_position(mockcontrol, FoMs, coordinates, expected):
    """Tests that the global minimum position is found and returned correctly"""
    constrained_par = Parameters([Parameter(name='parameter1', value=1., constraints=(0.5,2.0))])
    gpr = MinimizerFactory.create('GPR', mockcontrol(), constrained_par, n_points=3)
    min_coord, min_FoM = gpr.global_minimum_position(FoMs, coordinates)
    assert np.allclose(min_coord, expected[0], rtol=1e-5)
    assert np.allclose(min_FoM, expected[1], rtol=1e-5)

def test_GPR_create_bounds(mockcontrol):
    """Tests bounds are created and returned correctly"""
    constrained_parameter = Parameter(name='parameter1', value=1., constraints=(0.5,2.0))
    unconstrained_parameter = Parameter(name='parameter1', value=1.)
    unconstrained_parameter_zero = Parameter(name='parameter1', value=0.0)

    gpr = MinimizerFactory.create('GPR', mockcontrol(), Parameters(
        constrained_parameter), n_points=3)
    #  gpr needs to be instantiated, but isn't directly used
    lower_bound, upper_bound = gpr.create_bounds(constrained_parameter)
    assert np.allclose([lower_bound, upper_bound], [0.5,2.0], rtol=1e-5)

    lower_bound, upper_bound = gpr.create_bounds(unconstrained_parameter)
    assert np.allclose([lower_bound, upper_bound], [0.7,1.3], rtol=1e-5)

    with pytest.raises(ValueError):
        lower_bound, upper_bound = gpr.create_bounds(unconstrained_parameter_zero)


def test_GPR_set_parameter_values(mockcontrol):
    """Tests set_parameter_values can set values correctly"""
    constrained_par = Parameters([Parameter(name='parameter1', value=1., constraints=(0.5,2.0)),
                                 Parameter(name='parameter2', value=2., constraints=(0.3,6.0))])
    gpr = MinimizerFactory.create('GPR', mockcontrol(), constrained_par, n_points=3)
    gpr.set_parameter_values(['parameter1'], [1.9])
    assert gpr.parameters['parameter1'].value == 1.9

    gpr.set_parameter_values(['parameter1', 'parameter2'], [0.6, 1.56])
    assert gpr.parameters['parameter1'].value == 0.6
    assert gpr.parameters['parameter2'].value == 1.56

    with pytest.raises(ValueError):
        gpr.set_parameter_values(['parameter1'], [0.0])
    with pytest.raises(ValueError):
        gpr.set_parameter_values(['parameter2'], [7.0])


def test_GPR_fit(mockcontrol, parameters):
    """Tests that the GPR fit is called with the correct arguments given an input history"""
    mocked_df = pd.DataFrame(data=[[0, 100.0, 0.2, 2.6],
                            [1, 150.5, 1.8, 2.6]],
                            columns=['Unnamed: 0','FoM','epsilon','sigma'])

    with patch("MDMC.refinement.minimizers.GPR.pd.read_csv", autospec=True, return_value=mocked_df):
        with patch("MDMC.refinement.minimizers.GPR.skGPR.fit", autospec=True) as mock_fit:

            gpr = MinimizerFactory.create('GPR', mockcontrol(), parameters)
            _, _, _ = gpr.GPR_fit()
            # We don't care what the output is as not testing the scikit-learn module
            # we just want to know that it was called correctly.
            mock_fit.assert_called_with(ANY, [[0.2, 2.6], [1.8, 2.6]], [100.0, 150.5])


def test_GPR_predict(mockcontrol, parameters):
    """Tests that the GPR prediction returns the right points and predictions"""
    gpr = MinimizerFactory.create('GPR', mockcontrol(), parameters)
    kernel = RBF(length_scale=4.0)
    input_regressor = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
    input_regressor.fit([[0.0, 0.0], [1.0, 1.0]], [0.0, 1.0])
    min_params, min_fom = gpr.GPR_predict(input_regressor)
    assert np.allclose(min_params, [0.70000508, 2.59998984], rtol=1e-5)
    assert np.allclose(min_fom, 0.22619262570074802, rtol=1e-5)


def test_GPR_minimizer_change_constrained_parameter(mockcontrol):
    """Tests that constrained parameters do not exceed their max/min values."""
    parameters = Parameters([Parameter(name='constraints', value=1., constraints=(0.5, 1.5)),
                             Parameter(name='constraints_2', value=1., constraints=(0.5, 1.5))])

    # Expect values: The seed in the l.h.c. should make them consistent
    expected_values = [0.625, 1.375]
    gpr = MinimizerFactory.create('GPR', mockcontrol(), parameters)
    gpr.change_parameters()
    assert [p.value for p in gpr.parameters.values()] == expected_values


@pytest.mark.parametrize("has_converged_value",
                         [True, False])
def test_converge_message_in_output(GPR_with_history, mocked_df, has_converged_value):
    """Tests that the convergence message is present in the final output"""
    with patch("MDMC.refinement.minimizers.GPR.GPR.has_converged",
               autospec=True,
               return_value=has_converged_value):
        with patch("MDMC.refinement.minimizers.GPR.pd.read_csv",
                   autospec=True,
                   return_value=mocked_df):
            converged = GPR_with_history.has_converged()
            output_message = GPR_with_history.present_result()
            if converged:
                assert "The refinement has finished" in output_message
            else:
                assert "The refinement has not finished" in output_message


def test_GPR_FoM_and_coordinates_in_output(GPR_with_history, correct_output_data, mocked_df):
    """Tests that the correct coordinates present in the final output"""
    with patch("MDMC.refinement.minimizers.GPR.pd.read_csv", autospec=True, return_value=mocked_df):
        output_data = GPR_with_history.extract_result()
        output_string = GPR_with_history.format_result_string(output_data)

        assert str(correct_output_data[0]) in output_string
        assert str(correct_output_data[2]) in output_string

        assert np.allclose(correct_output_data[1], output_data[1], atol=0.0001, equal_nan=False)
        assert np.allclose(correct_output_data[3], output_data[3], atol=0.0001, equal_nan=False)
