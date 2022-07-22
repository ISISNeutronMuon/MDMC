"""
Tests the Minimizer base class
"""
import random
import re
from tempfile import NamedTemporaryFile

from unittest.mock import patch

import numpy as np
import pandas
import pytest

from MDMC.MD.parameters import Parameter, Parameters
from MDMC.refinement.minimizers.minimizer_factory import MinimizerFactory
from MDMC.refinement.minimizers.minimizer_abs import Minimizer

pytestmark = pytest.mark.mpi


@pytest.fixture
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

@patch.multiple(Minimizer, __abstractmethods__=set())
def test_minimizer_init(parameters):
    """
    Test initializing ``Minimizer``
    """
    # it's not worth parametrising a fixture just for this, so we use a loop
    cases = [parameters, list(parameters.values())]

    # Ignore pylint error as abstract class is mocked
    # pylint: disable=abstract-class-instantiated
    for parms in cases:
        minim = Minimizer(parms)
        assert np.all(minim.parameters == parameters)


@patch.multiple(Minimizer, __abstractmethods__=set())
def test_minimizer_init_invalid_parameters(parameters):
    """
    Test initializing ``Minimizer`` with fixed parameters, which should raise a
    `ValueError`
    """

    # Ignore pylint error as abstract class is mocked
    # pylint: disable=abstract-class-instantiated
    fixed_parameter = Parameter(name='fixed_parameter', value=1.)
    fixed_parameter.fixed = True
    parameters.append(fixed_parameter)

    with pytest.raises(ValueError):
        Minimizer(parameters)


@patch.multiple(Minimizer, __abstractmethods__=set())
def test_minimizer_write_history(parameters):
    """
    Test history csv output of ``Minimizer``
    """
    class MockMinimizer(Minimizer):

        @property
        def history_columns(self):
            return ['A', 'B', 'C']

    # Ignore pylint error as abstract class is mocked
    # pylint: disable=abstract-class-instantiated
    minim = MockMinimizer(parameters)
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
def test_minimizer_history_columns(parameters, p_slice, columns):
    """
    Tests that the history columns for the` minimizer are as expected,
    including the names of the ``Parameter`` objects which are refined
    """
    parameter_slice = Parameters(list(parameters.values())[slice(*p_slice)])

    for minimizer_name in MinimizerFactory.get_minimizer_names():
        minim = MinimizerFactory.create_minimizer(minimizer_name, parameter_slice)
        expected_columns = columns
        expected_columns.extend(['FoM', 'Change state'])

        for expected_column in expected_columns:
            assert np.any([expected_column in history_columns for \
                            history_columns in minim.history_columns])

def test_minimizer_fixed_parameter():
    """
    Test that a ``ValueError`` is raised when passing a fixed ``Parameter``
    """

    parameters = [Parameter(name='fixed', value=1., fixed=True)]
    with pytest.raises(ValueError):
        for minimizer_name in MinimizerFactory.get_minimizer_names():
            _ = MinimizerFactory.create_minimizer(minimizer_name, parameters)


def test_minimizer_tied_parameter():
    """
    Test that a ``ValueError`` is raised when passing a tied ``Parameter``
    """
    target_parameter = Parameter(name='target', value=1., )
    tied_parameter = Parameter(name='tied', value=1., )
    tied_parameter.set_tie(target_parameter, ' * 2')
    parameters = [tied_parameter]
    with pytest.raises(ValueError):
        for minimizer_name in MinimizerFactory.get_minimizer_names():
            minim = MinimizerFactory.create_minimizer(minimizer_name, parameters)


@patch.multiple(Minimizer, __abstractmethods__=set())
@pytest.mark.parametrize("params_last, FoM_last, params_lowest, FoM_lowest",
                         [
                             ((1.021885, 3.403754), 419.142104, (1.026101, 3.381142), 405.601993),
                             ((2.82347, 5.238947), 300., (2.82347, 5.238947), 300.),
                             ((2134, 12344), 23456., (42343., 342.), 1034.)
                          ])
def test_correct_output(parameters, params_last, FoM_last, params_lowest, FoM_lowest):
    """
    Test that the output of a minimizer is returned in the correct format
    """
    class MockMinimizer(Minimizer):
        def has_converged(self):
            return True if params_last == params_lowest else False

    minimizer = MockMinimizer(parameters)
    obtained_output_string = minimizer.format_result_string(
        params_last,
        FoM_last,
        params_lowest,
        FoM_lowest)

    converged_message = "\nThe refinement has converged." if minimizer.has_converged() else "\nThe refinement has not converged."

    expected_output_string = (f'{converged_message} \n \n'
                                 f'Last accepted point is: \n'
                                 f'{params_last} with a minimum '
                                 f'FoM of {FoM_last}. \n \n'
                                 f'Best point measured was: \n'
                                 f'{params_lowest} for a minimum FoM of '
                                 f'{FoM_lowest}.\n \n ')

    assert obtained_output_string == expected_output_string


@patch.multiple(Minimizer, __abstractmethods__=set())
@pytest.mark.parametrize("params_last, FoM_last, params_lowest, FoM_lowest",
                         [
                             ("abc", ["Wrong", "Value"], (1.026101, 3.381142), 405.601993),
                             (2, {"Wrong": 123}, False, ("",)),
                             ((1.309348, 2.87394, 10.3489), 456., (123, 23, 42), "FoM")
                          ])
def test_incorrect_input_for_output_string(parameters, params_last, FoM_last, params_lowest, FoM_lowest):
    """
    Test that a TypeError is raised if the wrong input type is provided to format_result_string
    """
    class MockMinimizer(Minimizer):
        def has_converged(self):
            return False

    minimizer = MockMinimizer(parameters)
    with pytest.raises(TypeError):
        print(minimizer.format_result_string(params_last, FoM_last, params_lowest, FoM_lowest))


def test_each_minimizer_for_correct_output(parameters):
    """
    Tests each implemented minimizer to make sure that the output is given in the same format
    """
    for minimizer_name in MinimizerFactory.get_minimizer_names():
        minimizer = MinimizerFactory.create_minimizer(minimizer_name, parameters)

        # It does not matter what FoM is - we just want some history to check the output
        randomizer = random.Random()
        for i in range(5):
            minimizer.step(FoM=randomizer.uniform(0.1, 1000))
        obtained_history_string = minimizer.present_result()
        pattern = re.compile(r"\nThe refinement has not converged\. \n \nLast accepted point is: \n\(.*\) with a "
                             r"minimum FoM of .*\..*\. \n \nBest point measured was: \n\(.*\) for a minimum FoM of "
                             r".*\..*\.\n \n")

        assert re.match(pattern, obtained_history_string) is not None
