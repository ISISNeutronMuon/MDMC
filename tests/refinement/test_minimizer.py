"""Tests the Minimizer base class"""
from tempfile import NamedTemporaryFile
from copy import copy

from unittest.mock import patch

from pathlib import Path
import numpy as np
import pytest
import pandas as pd

from MDMC.MD.parameters import Parameter, Parameters
from MDMC.refinement.minimizers.minimizer_factory import MinimizerFactory
from MDMC.refinement.minimizers.minimizer_abs import Minimizer

pytestmark = pytest.mark.mpi


class MockControl:

    def __init__(self):
        self.n_steps = 20


@pytest.fixture(scope="module")
def mockcontrol():

    _mockcontrol = MockControl()
    return _mockcontrol

@pytest.fixture(scope="module")
def parameters():
    """
    Returns
    -------
    Parameters
        A `dict-like` of ``Parameter`` objects with a variety of `name` and
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


@pytest.mark.skip
def remove_fixed_parameter(params_obj):
    for param_name in params_obj.keys():
        if param_name.startswith("fixed_parameter"):
            params_obj.pop(param_name)
            break

@patch.multiple(Minimizer, __abstractmethods__=set())
def test_minimizer_init(mockcontrol, parameters):
    """Test initializing ``Minimizer``"""
    # it's not worth parametrising a fixture just for this, so we use a loop
    cases = [parameters, list(parameters.values())]

    # Ignore pylint error as abstract class is mocked
    # pylint: disable=abstract-class-instantiated
    for parms in cases:
        minim = Minimizer(mockcontrol, parms)
        assert np.all(minim.parameters == parameters)


@patch.multiple(Minimizer, __abstractmethods__=set())
def test_minimizer_init_invalid_parameters(mockcontrol, parameters):
    """Test initializing ``Minimizer`` with fixed parameters, which should raise a `ValueError`"""

    # Ignore pylint error as abstract class is mocked
    # pylint: disable=abstract-class-instantiated
    fixed_parameter = Parameter(name='fixed_parameter', value=1.)
    fixed_parameter.fixed = True
    parameters.append(fixed_parameter)

    with pytest.raises(ValueError):
        Minimizer(mockcontrol, parameters)


@patch.multiple(Minimizer, __abstractmethods__=set())
def test_minimizer_write_history(mockcontrol, parameters):
    """Test history csv output of ``Minimizer``"""
    class MockMinimizer(Minimizer):

        @property
        def history_columns(self):
            return ['A', 'B', 'C']

    # Ignore pylint error as abstract class is mocked
    # pylint: disable=abstract-class-instantiated

    remove_fixed_parameter(parameters)

    minim = MockMinimizer(mockcontrol, parameters)
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
                           ['A', 'B', 'C', 'charge']),
                          ([0, 9, 2],
                           ['A', 'C', 'charge', 'equilibrium_state', 'sigma'])
                          ])
@pytest.mark.parametrize('minimizer_name', MinimizerFactory.get_minimizer_names())
def test_minimizer_history_columns(mockcontrol, parameters, p_slice, columns, minimizer_name):
    """
    Tests that the history columns for the` minimizer are as expected,
    including the names of the ``Parameter`` objects which are refined
    """
    parameter_slice = Parameters(list(parameters.values())[slice(*p_slice)])

    minim = MinimizerFactory.create_minimizer(minimizer_name, mockcontrol, parameter_slice)
    expected_columns = copy(columns)  # copy required else the actual parameters get changed by append
    expected_columns.append('FoM')
    if minimizer_name == "MMC":  # only MMC has the 'accept'/'reject' column
        expected_columns.append('Change state')

    for expected_column in expected_columns:
        assert np.any([expected_column in history_columns for \
                        history_columns in minim.history_columns])


def test_minimizer_fixed_parameter():
    """Test that a ``ValueError`` is raised when passing a fixed ``Parameter``"""

    parameters = [Parameter(name='fixed', value=1., fixed=True)]
    with pytest.raises(ValueError):
        for minimizer_name in MinimizerFactory.get_minimizer_names():
            _ = MinimizerFactory.create_minimizer(minimizer_name, mockcontrol, parameters)


def test_minimizer_tied_parameter():
    """Test that a ``ValueError`` is raised when passing a tied ``Parameter``"""
    target_parameter = Parameter(name='target', value=1., )
    tied_parameter = Parameter(name='tied', value=1., )
    tied_parameter.set_tie(target_parameter, ' * 2')
    parameters = [tied_parameter]
    with pytest.raises(ValueError):
        for minimizer_name in MinimizerFactory.get_minimizer_names():
            minim = MinimizerFactory.create_minimizer(minimizer_name, mockcontrol, parameters)

@patch.multiple(Minimizer, __abstractmethods__=set())    
@pytest.mark.parametrize('column_names, previous_history'
                        ,[(['param 1', 'Change state'], [[1,'Accepted'],[2, 'Rejected']])
                        ,(['param 1', 'param 2'], [[1.0,2.0],[3.14,4.00]])])
def test_load_history(mockcontrol, parameters,column_names, previous_history):
    """Test that loading a previous refinement file loads successfully with different
    types of data."""
    
    minim = Minimizer(mockcontrol, parameters)
    df = pd.DataFrame(previous_history, columns=column_names)
    temp_file = NamedTemporaryFile()
    df.to_csv(temp_file.name)
    
    params, _ = minim.load_history(Path(temp_file.name))
    assert list(column_names) == list(params)

 
@patch.multiple(Minimizer, __abstractmethods__=set())    
@pytest.mark.parametrize("column_names,history",
                        [(['FoM','dummy (#1)', 'dummy (#2)'],[[1,2],[3,4]])
                         ,(['FoM','param1 (#2)', 'param1 (#3)'], [[1,2],[3,4]]) 
                        ,(['FoM','param (#1)', 'param (#2)', 'param (#3)'], [[1,2], [3,4]])])
def test_check_parameters_fit_with_history(mockcontrol, parameters,column_names, history):
    """Test that an Exception is raised when the parameters, or data type, in the previous refinement file
    are not compatible with the ones currently defined in the control object 
    (for example, different name, different number of parameters)."""
    
    
    parameters = [Parameter(1.0, 'param'), Parameter(2.0, 'param')]
    parameters = Parameters(parameters)
    minim = Minimizer(mockcontrol, parameters)

    with pytest.raises(Exception):
        minim._check_parameters_fit_with_history(parameters, column_names, history)

@patch.multiple(Minimizer, __abstractmethods__=set())    
@pytest.mark.parametrize("column_names, history", [([],[[1.0,2.0],[3.0,4.0], [5.0,6.0]])])
def test_get_parameters_old_values(mockcontrol,column_names, history):
    """Test that the previous parameter values, as loaded in from the previous refinement file, 
    are retrieved correctly"""
    
    parameters = [Parameter(1.0, 'param'), Parameter(2.0, 'param')]
    parameters = Parameters(parameters)
    minim = Minimizer(mockcontrol, parameters)
    column_names = [param for param in parameters]
    minim.get_parameters_old_values(parameters, column_names, history)

@patch.multiple(Minimizer, __abstractmethods__=set())
@pytest.mark.parametrize("column_names, history",
    [(['FoM', 'Change state'], [[10.0, 1.0,2.0], [15.0, 2.0,3.0]])
     ,(['Fom'], [[10.0, 'Accepted', 1.0, 2.0],[15.0, 'Rejected', 2.0,3.0]])])
def test_different_minimizer_compatibility(mockcontrol, column_names, history):
    """
    Test that a file with data generated using a different minimizer to the one specified with the
    current set up can be used and is compatible."""
    
    parameters = [Parameter(1.0, 'param'), Parameter(2.0, 'param')]
    parameters = Parameters(parameters)
    for param in parameters:
        column_names.append(str(param))
    
    minim = Minimizer(mockcontrol,parameters)
    minim.different_minimizer_compatibility(column_names, history)
