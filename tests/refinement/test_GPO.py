"""
Tests the GPO minimizer class
"""

import numpy as np
import pytest

from MDMC.refinement.minimizers.GPO import GPO
from MDMC.refinement.minimizers.minimizer_abs import Minimizer
from MDMC.refinement.minimizers.minimizer_factory import MinimizerFactory
from MDMC.MD.parameters import Parameters, Parameter

@pytest.fixture
def parameters() -> Parameters:
    """
    A fixture returning two arbitrary `Parameter` objects wrapped in a `Parameters` collection.
    """
    return Parameters([Parameter(name='parameter1', value=1.),
                Parameter(name='parameter2', value=2.)])


@pytest.fixture
def constrained_parameters() -> Parameters:
    """
    A fixture returning two arbitrary `Parameter` objects with constraints on their values,
    wrapped in a `Parameters` collection.
    """
    return Parameters([Parameter(name='parameter1', value=1., constraints=(0.5, 2.0)),
                Parameter(name='parameter2', value=2., constraints=(1.0, 4.0))])


@pytest.fixture
def GPO_with_history(parameters) -> Minimizer:
    """
    Creates an instance of GPO with a 10-step history

    Returns
    -------
        A GPO object with a history of 10 steps
    """

    minimizer = GPO(parameters)
    for i in range(10):
        minimizer.step(FoM=i)
    return minimizer


@pytest.mark.skip
def obtain_correct_output_values(GPO_obj):
    """
    A function to obtain the correct values from a GPOs history
    """
    FoMs = []
    for FoM in GPO_obj.history.values:
        FoMs.append(FoM[:][0])
    min_FoM_measured = np.min(FoMs)
    min_parameters_measured = GPO_obj.history.values[np.where(FoMs == min_FoM_measured)[0][0]][3]
    # the [0][0][3] is to get the parameters from the _history
    predicted_min_pos = tuple(GPO_obj.predicted_min_pos)
    min_parameters_measured = tuple([min_parameters_measured])

    list_of_outputs = [
        min_parameters_measured,
        min_FoM_measured,
        predicted_min_pos,
        GPO_obj.predicted_FoM
    ]
    return list_of_outputs

@pytest.mark.parametrize('mock_history, min_steps, expected',
                         [([[3, 'Accepted', 1, 1, 4], [2, 'Accepted', 1, 1, 3], [2, 'Accepted', 1, 1, 3]], 4, False),
                          ([[3, 'Accepted', 1, 1, 4], [2, 'Accepted', 1, 1, 3], [2, 'Accepted', 1, 1, 3]], 3, True),
                          ([[2, 'Accepted', 1, 1, 3], [2, 'Accepted', 1, 1, 3], [2, 'Accepted', 1, 1, 3]], 2, True)])
def test_GPO_has_converged(mock_history, min_steps, expected):
    """ Test that the array of points to be simulated is created correctly """
    parameter = Parameters(Parameter(name='A', value=1))
    gpo = MinimizerFactory.create_minimizer('GPO', parameter, n_points=min_steps)
    gpo._history = mock_history
    assert gpo.has_converged() == expected


def test_GPO_step():
    """ Tests GPO is able to find the minima of a single cycle of a cosine function """
    parameter = Parameters(Parameter(name='a', value=1.5, constraints=[-2., 4.]))
    gpo = MinimizerFactory.create_minimizer('GPO', parameter, n_points=100)
    gpo._history=[]
    for _ in range(25):
        x = parameter['a'].value
        FoM=np.cos(x)+3.0
        gpo.step(FoM)
    assert np.allclose([gpo.predicted_min_pos], [np.pi], atol=1e-2)


def test_GPO_set_parameter_values():
    """Tests set_parameter_values can set values correctly"""
    constrained_par = Parameters([Parameter(name='parameter1', value=1., constraints=(0.5,2.0)),
                                 Parameter(name='parameter2', value=2., constraints=(0.3,6.0))])
    gpo = MinimizerFactory.create_minimizer('GPO', constrained_par, n_points=3)
    gpo.set_parameter_values(['parameter1'], [1.9])
    assert gpo.parameters['parameter1'].value == 1.9

    gpo.set_parameter_values(['parameter1', 'parameter2'], [0.6, 1.56])
    assert gpo.parameters['parameter1'].value == 0.6
    assert gpo.parameters['parameter2'].value == 1.56

    with pytest.raises(ValueError):
        gpo.set_parameter_values(['parameter1'], [0.0])
    with pytest.raises(ValueError):
        gpo.set_parameter_values(['parameter2'], [7.0])


def test_converge_message_in_output(GPO_with_history):
    """ Tests that the convergence message is present in the final output """

    converged = GPO_with_history.has_converged()
    output_message = GPO_with_history.present_result()
    if converged:
        assert "The refinement has finished" in output_message
    else:
        assert "The refinement has not finished" in output_message


def test_GPO_FoM_and_coordinates_in_output(GPO_with_history):
    """ Tests that the correct values are  present in the final output """
    output_data = GPO_with_history.extract_result()
    output_string = GPO_with_history.format_result_string(output_data)
    expected_data = obtain_correct_output_values(GPO_with_history)

    assert str(expected_data[0]) in output_string
    assert str(expected_data[2]) in output_string

    assert np.allclose(expected_data[1], output_data[1], atol=0.0001, equal_nan=False)
    assert np.allclose(expected_data[3], output_data[3], atol=0.0001, equal_nan=False)
