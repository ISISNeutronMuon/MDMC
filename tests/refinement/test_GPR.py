"""
Tests the GPR minimizer class
"""

from unittest.mock import patch, ANY, PropertyMock

import numpy as np
import pandas
import pandas as pd
import pytest

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from MDMC.MD.parameters import Parameter, Parameters
from MDMC.refinement.minimizers.minimizer_factory import MinimizerFactory


def test_GPR_parameter_point_array():
    """Test that the array of points to be simulated is created correctly"""
    parameters = Parameters([Parameter(name='parameter1', value=1.),
                             Parameter(name='parameter2', value=2.)])
    gpr = MinimizerFactory.create_minimizer('GPR', parameters, n_points=2)
    points = gpr.parameter_point_array
    assert np.allclose(points[0], (0.8, 1.6), rtol=1e-5)
    assert np.allclose(points[1], (0.8, 2.4), rtol=1e-5)
    assert np.allclose(points[2], (1.2, 1.6), rtol=1e-5)
    assert np.allclose(points[3], (1.2, 2.4), rtol=1e-5)

    constrained_pars = Parameters([Parameter(name='parameter1', value=1., constraints=(0.5,2.0)),
                                   Parameter(name='parameter2', value=2.,  constraints=(1.0,4.0))])
    _, points = gpr.create_parameter_point_array(constrained_pars)
    assert np.allclose(points[0], [0.5, 1.0], rtol=1e-5)
    assert np.allclose(points[1], [0.5, 4.0], rtol=1e-5)
    assert np.allclose(points[2], [2.0, 1.0], rtol=1e-5)
    assert np.allclose(points[3], [2.0, 4.0], rtol=1e-5)

def test_GPR_parameter_point_array_hypercube():
    """Test that the array of points to be simulated is created correctly for the latin hypercube"""
    constrained_pars = Parameters([Parameter(name='parameter1', value=1., constraints=(0.5,2.0)),
                                   Parameter(name='parameter2', value=2.,  constraints=(1.0,4.0))])
    gpr = MinimizerFactory.create_minimizer('GPR', constrained_pars, n_points=4, use_hypercube=True)
    points = gpr.parameter_point_array
    par1_constraints = constrained_pars['parameter1'].constraints
    par2_constraints = constrained_pars['parameter2'].constraints

    assert len(points) == 4
    assert np.all([np.array(points)[:,0]>=par1_constraints[0],
                   np.array(points)[:,0]<=par1_constraints[1]])

    assert np.all([np.array(points)[:,1]>=par2_constraints[0],
                   np.array(points)[:,1]<=par2_constraints[1]])


def test_GPR_reset_parameters():
    """Test that parameters get reset"""
    parameters = Parameters([Parameter(name='parameter1', value=1.),
                Parameter(name='parameter2', value=2.)])
    gpr = MinimizerFactory.create_minimizer('GPR', parameters, n_points=2)

    parameter_values = [p.value for p in gpr.parameters.values()]
    assert np.allclose(parameter_values, (0.8, 1.6), rtol=1e-5)

    gpr.reset_parameters()
    parameter_values = [p.value for p in gpr.parameters.values()]
    assert np.allclose(parameter_values, (1.2, 2.4), rtol=1e-5)

@pytest.mark.parametrize('FoMs,coordinates,expected',
    [([2, 3, 0, 1, 4], [[0,0], [0,1], [1,0], [1,1], [2,0]], [[1,0], 0]),
    ([2], [[0,0,1]], [[0,0,1], 2]),
    ([0.01, 0.020, 0.01, 6], [[0.1,0.1,0.1],[0.1,0.1,1],[0.1,1,1],[1,1,1]], [[0.1,0.1,0.1], 0.01])])
def test_GPR_global_minimum_position(FoMs, coordinates, expected):
    """Tests that the global minimum position is found and returned correctly"""
    constrained_par = Parameters([Parameter(name='parameter1', value=1., constraints=(0.5,2.0))])
    gpr = MinimizerFactory.create_minimizer('GPR', constrained_par, n_points=3)
    min_coord, min_FoM = gpr.global_minimum_position(FoMs, coordinates)
    assert np.allclose(min_coord, expected[0], rtol=1e-5)
    assert np.allclose(min_FoM, expected[1], rtol=1e-5)

def test_GPR_create_bounds():
    """Tests bounds are created and returned correctly"""
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
    """Tests set_parameter_values can set values correctly"""
    constrained_par = Parameters([Parameter(name='parameter1', value=1., constraints=(0.5,2.0)),
                                 Parameter(name='parameter2', value=2., constraints=(0.3,6.0))])
    gpr = MinimizerFactory.create_minimizer('GPR', constrained_par, n_points=3)
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
    """Tests that the GPR fit is called with the correct arguments given an input history"""
    mocked_df = pd.DataFrame(data=[[0,100.0,'Accepted',0.2,2.6],
                            [1,150.5,'Accepted',1.8,2.6]],
                            columns=['Unnamed: 0','FoM','Change state','epsilon','sigma'])

    with patch("MDMC.refinement.minimizers.GPR.pd.read_csv", autospec=True, return_value=mocked_df):
        with patch("MDMC.refinement.minimizers.GPR.skGPR.fit", autospec=True) as mock_fit:

            gpr = MinimizerFactory.create_minimizer('GPR', Parameters())
            _, _, _ = gpr.GPR_fit()
            # We don't care what the output is as not testing the scikit-learn module
            # we just want to know that it was called correctly.
            mock_fit.assert_called_with(ANY, [[0.2, 2.6], [1.8, 2.6]], [100.0, 150.5])

def test_GPR_predict():
    """Tests that the GPR prediction returns the right points and predictions"""
    gpr = MinimizerFactory.create_minimizer('GPR', Parameters())
    kernel = RBF(length_scale = 4.0)
    input_regressor = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
    input_regressor.fit([[0.0, 0.0], [1.0, 1.0]], [0.0, 1.0])
    point_array, prediction = gpr.GPR_predict(input_regressor, points=2)
    assert np.allclose(point_array, [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], rtol=1e-5)
    assert np.allclose(prediction, [0.03015209, 0.40226347, 0.40226347, 0.89999947], rtol=1e-5)

def test_GPR_minimizer_change_constrained_parameter():
    """
    Tests that constrained parameters do not exceed their max/min values.
    """
    parameters = Parameters([Parameter(name='constraints', value=1., constraints=(0.5, 1.5)),
                             Parameter(name='constraints_2', value=1., constraints=(0.5, 1.5))])

    # Expect values to be set to the upper/lower limit
    expected_values = [0.5, 0.5]
    minim = MinimizerFactory.create_minimizer('GPR', parameters)
    minim.change_parameters()
    assert [p.value for p in minim.parameters.values()] == expected_values

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
def test_GPR_present_results(mock_history, FoMs, expected):
    """
    Tests that the output of GPR contains the correct refined coordinates.
    """

    params = Parameters()
    with patch("MDMC.refinement.minimizers.GPR.GPR.history", new_callable=PropertyMock) as hist:
        hist.return_value = mock_history
        with patch("MDMC.refinement.minimizers.GPR.GPR.history_columns", new_callable=PropertyMock) as columns:
            columns.return_value = list(mock_history.columns)
            gpr = MinimizerFactory().create_minimizer("GPR", params)
            output_string = gpr.present_result()
            assert str(FoMs[0]) in output_string
            assert str(FoMs[1]) in output_string
            assert str(expected[0]) in output_string
            assert str(expected[1]) in output_string