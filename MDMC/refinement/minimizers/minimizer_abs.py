"""A module for all minimizers which can be iterated to refine the potential
parameters"""
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import csv
import numpy as np

from MDMC.MD import Parameters
from MDMC.common.decorators import repr_decorator


if TYPE_CHECKING:
    from MDMC.control import Control


@repr_decorator('FoM', 'FoM_old',
                'parameters', 'parameters_old_values')
class Minimizer(ABC):

    """
    An abstract class with methods common to all minimizers

    Parameters
    ----------
    control : Control
        The ``Control`` object which uses this Minimizer.
    parameters : Parameters or list of Parameter
        A `list` of ``Parameter`` objects which will be fit

    Attributes
    ----------
    history : list
        A `list` of minimization history, where each element contains the FoM, a
        `list` of the ``Parameters`` and a `str` with whether the step was
        Accepted or Rejected.
    FoM : float
        The FoM from the current ``Minimizer`` step
    FoM_old : float
        The FoM from the previous ``Minimizer`` step
    parameters : Parameters
        A ``Parameters`` object containing the ``Parameter`` objects being fitted
    parameters_old_values : Parameters
        A ``Parameters`` object containing the values of
        the ``Parameter`` objects from the previous minimizer step
    """

    def __init__(self, control: 'Control', parameters: Parameters, previous_history: Path = None):
        self.control = control
        self.results_filename = None
        self.previous_history = previous_history
        self.previous_steps = 0

        if previous_history is not None:
            if not isinstance(previous_history, str):
                self.previous_history = Path(self.previous_history)
                self._history = self.load_history(self.previous_history)
            else:
                self._history = self.load_history(self.previous_history)
                
            self.previous_steps = len(self._history)
                
            self.FoM_old = self._history[-1][0]
            self.FoM = None
            
            if isinstance(parameters, list):
                parameters = Parameters(parameters)
            
            self._check_parameters_fit_with_history(parameters)
            self._check_parameters(parameters)
            self.parameters_old_values = self.get_parameters_old_values(parameters)
            self.parameters = parameters
 
        else:
            self._history = []
            self.FoM_old = float('inf')
            self.FoM = None
            if isinstance(parameters, list):
                parameters = Parameters(parameters)
            self._check_parameters(parameters)
            self.parameters_old_values = None
            self.parameters = parameters

    @abstractmethod
    def step(self, FoM: float) -> None:
        """
        Increments the minimization by a step

        Parameters
        ----------
        FoM : float
            The current figure of merit value.
        """

        raise NotImplementedError

    @property
    def history(self) -> pd.DataFrame:
        """
        Get the history of the minimizer, with a single entry for each step of
        the minimizer

        Returns
        -------
        pd.DataFrame
            Contains the minimizer variables for each refinement step. The
            variables which are included is concrete implementation specific,
            and is specified by `history_columns`.
        """

        return pd.DataFrame(self._history, columns = self.history_columns)

    @property
    @abstractmethod
    def history_columns(self) -> 'list[str]':
        """
        Get the column titles for the minimizer history

        Returns
        -------
        list
            A 'list' of 'str' specifying the column titles for the minimizer
            history
        """

        raise NotImplementedError

    @abstractmethod
    def change_parameters(self) -> None:
        """Selects a new value for each ``Parameter``."""

        raise NotImplementedError

    @abstractmethod
    def has_converged(self) -> bool:
        """
        Checks if the refinement process has converged/finished. The condition
        which needs to be met to make this True is optimizer dependent, but
        might be that the refinement has repeatedly returned a very similar FoM
        which meets some threshold, determining that it is close to the optimal,
        or it could be that the minimizer has measured at all the parameter points
        that were specified and it should now predict the best position.

        Returns
        -------
        bool
            Whether or not the minimizer has converged/finished.
        """

        raise NotImplementedError

    @staticmethod
    def _check_parameters(parameters: Parameters) -> None:
        """
        Checks the validity of the parameters on input

        Parameters
        ----------
        parameters : Parameters
            All ``Parameter`` objects to validate

        Raises
        ------
        ValueError
            If any ``Parameter`` is fixed
            If there is mismatch with previous value in history.
        """

        for parameter in parameters.values():
            if parameter.fixed is True:
                raise ValueError(
                    f'Parameter {parameter.name} is fixed, and so cannot be refined')
            if parameter.tied is True:
                raise ValueError(f'Parameter {parameter.name} is tied to the value of '
                                 'another parameter and so cannot be refined')

    def write_history(self, filename) -> None:
        """
        Write the minimizer history to a csv file

        Parameters
        ----------
        filename : str
            The name of the output file
        """

        self.history.to_csv(filename)

    def present_result(self) -> str:
        """
        Extracts and returns the most appropriate output for the
        minimiser class, in an appropriate format
        e.g. minimum FOM and parameter values

        Returns
        -------
        str
            A formatted string representing the output of the minimizer
        """
        extracted_results = self.extract_result()
        return self.format_result_string(extracted_results)

    @abstractmethod
    def reset_parameters(self) -> None:
        """Resets the parameters to a previous state"""

        raise NotImplementedError

    @abstractmethod
    def extract_result(self) -> 'list[str]':
        """
        Obtains the result of the minimizer to be presented/formatted

        Returns
        -------
        list[str]
            A list of strings representing the data points
            output by the minimizer to be formatted into a string
        """
        raise NotImplementedError

    @abstractmethod
    def format_result_string(self, minimizer_output: list) -> str:
        """
        Formats a string output for the results of the minimiser class.

        Parameters
        ----------
        minimizer_output: list
            A list of printable values representing the data points
            output by the minimizer to be formatted into a string

        Returns
        -------
        str
            A string encompassing the output of the minimizer.
        """
        raise NotImplementedError


    def load_history(self, file_path: Path):
        try:
            with open(self.previous_history, 'r') as file:
                file_content = np.array(list(csv.reader(file)))
        except ValueError:
            raise ValueError("Can not find file or path.")

        # remove empty index and separate column names
        file_content = [row[1:] for row in file_content]
        self.column_names = file_content[0]
        del file_content[0]
        
        file_content = [np.asfarray(row) for row in file_content]

        return (file_content)


    def _check_parameters_fit_with_history(self, parameters: Parameters) -> bool:
        if self._history is not None:
            # using a reduced length for 'column_names' because it includes 'FoM' 
            # and we want parameters only.
            if (len(self.column_names)-1) != len(parameters):
                raise ValueError(f'A history of {len(self._history.columns) -2}'\
                    ' is incompatible with the current setup.')
            
            split_param_list = [parameter.split(" ")[0] for parameter in parameters]
            split_column_list = [column.split(" ")[0] for column in self.column_names[1:]]
            
            if split_param_list != split_column_list:
                raise ValueError(f"The parameters in the minimizer history are not \
                                      the same as those specified for refining in the current\
                                      universe setup.")
            param_list = [parameter for parameter in parameters]
            param_list.insert(0, 'Fom')
            self.column_names = param_list
            
            # for parameter in parameters.values():
            #     if parameter.fixed is True:
            #         raise ValueError(f'Parameter {parameter.name} is fixed.')
            #     if parameter.tied is True:
            #         raise ValueError(f'Parameter {parameter.name} is tied to another parameter.')
            #     # if parameter.name in self._history.columns and type(parameter.value) != type(self._history[parameter.name].iloc[-1]):
            #     #     raise ValueError(f"Parameter {parameter.name} has a type mismatch with the last recorded value in history.")
            return True
                

    def get_parameters_old_values(self, parameters: Parameters):
        if self._history:
            last_entry = self._history[-1]
            # old_values = {param.name: last_entry[param.name] for param in parameters.values()}
            old_values = {}
            for param in parameters.values():
                pos  = self.column_names.index(param.name)
                old_values[param.name] = last_entry[pos]
            return old_values
        else:
            return None