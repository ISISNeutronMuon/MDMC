"""
Tests the Minimizer base class
"""

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
def test_correct_output_with_numpy_data(parameters):
    """
    Test that the output of a minimizer is in the correct format,
    using a pandas DataFrame as the history
    """

    class DataFrameHistoryMinimizer(Minimizer):

        @property
        def history_columns(self):
            return ['FoM', 'Change state', 'epsilon (#2)', 'sigma (#3)']

        @property
        def history(self):
            return pandas.DataFrame(
                data=[
                [466.498806, "Accepted", 1.024300, 3.360000],
                [439.172624, "Accepted", 1.018742, 3.384798],
                [405.601993, "Accepted", 1.026101, 3.381142],
                [462.436135, "Rejected", 1.020707, 3.369637],
                [456.283837, "Rejected", 1.035037, 3.397473],
                [419.142104, "Accepted", 1.021885, 3.403754]
                ],
                columns=self.history_columns
            )

        def has_converged(self):
            return False

    mini = DataFrameHistoryMinimizer(parameters)
    obtained_output_string = mini.present_result()
    expected_output_string = (f'\nThe refinement has not converged. \n \n'
                         f'Last accepted point is: \n'
                         f'(1.021885, 3.403754) with a minimum '
                         f'FoM of 419.142104. \n \n'
                         f'Best point measured was: \n'
                         f'(1.026101, 3.381142) for a minimum FoM of '
                         f'405.601993.\n \n ')

    assert obtained_output_string == expected_output_string

    @patch.multiple(Minimizer, __abstractmethods__=set())
    def test_correct_output_with_list_like_data(parameters):
        """
        Test that the output of a minimizer is in the correct format,
        using a 2D List as the history
        """

        class ListHistoryMinimizer(Minimizer):

            @property
            def history_columns(self):
                return ['FoM', 'Change state', 'epsilon (#2)', 'sigma (#3)']

            @property
            def history(self):
                return [
                    [466.498806, "Accepted", 1.024300, 3.360000],
                    [439.172624, "Accepted", 1.018742, 3.384798],
                    [405.601993, "Accepted", 1.026101, 3.381142],
                    [462.436135, "Rejected", 1.020707, 3.369637],
                    [456.283837, "Rejected", 1.035037, 3.397473],
                    [419.142104, "Accepted", 1.021885, 3.403754]
                ]

            def has_converged(self):
                return False

        mini = ListHistoryMinimizer(parameters)
        obtained_output_string = mini.present_result()
        expected_output_string = (f'\nThe refinement has not converged. \n \n'
                                  f'Last accepted point is: \n'
                                  f'(1.021885, 3.403754) with a minimum '
                                  f'FoM of 419.142104. \n \n'
                                  f'Best point measured was: \n'
                                  f'(1.026101, 3.381142) for a minimum FoM of '
                                  f'405.601993.\n \n ')

        assert obtained_output_string == expected_output_string