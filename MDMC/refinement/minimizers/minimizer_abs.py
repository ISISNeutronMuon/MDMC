"""A module for all minimizers which can be iterated to refine the potential
parameters"""
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
import csv
from pathlib import Path
import pandas as pd

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
        self.compatible = False

        if isinstance(parameters, list):
            parameters = Parameters(parameters)
        self.parameters = parameters

        if previous_history:
            if isinstance(previous_history, str):
                self.previous_history = Path(self.previous_history)

            self.column_names, self._history = \
            self.load_history(self.previous_history)
            self.previous_steps = len(self._history)
            self.FoM_old = self._history[-1][0]
            self.FoM = None
            self.different_minimizer_compatibility(self.history_columns, self._history)
            self._check_parameters_fit_with_history(parameters, self.column_names, self._history)
            self.parameters_old_values = self.get_parameters_old_values(parameters, \
                self.column_names, self._history)
        else:
            self._history = []
            self.FoM_old = float('inf')
            self.parameters_old_values = None

        self._check_parameters(parameters)
        self.FoM = None

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


    def load_history(self, history: Path) -> tuple:
        """Uses the `previous_history` variable to load a file of previous refinement steps.
        It then formats this into the column names and the actual parameter values. The loaded data
        is stored as numpy arrays.

        Parameters
        ----------
        history: Path
            A file path which contains previous refinement data.

        Raises
        ----------
        ValueError
            If the file with the previous history data can not be found.

        Returns
        ----------
        tuple
            list of columns names, and a list containing a list for each refinement step
             from the loaded history file."""
        try:
            with open(history, 'r', encoding='utf-8') as file:
                file_content = list(csv.reader(file))
        except ValueError as err:
            raise ValueError("Can not find file or path.") from err

        # remove empty index and separate column names
        file_content = [row[1:] for row in file_content]
        column_names = file_content[0]
        del file_content[0]
        file_content = ([[float(x) if x.isdigit() or x.replace(".","").isnumeric() \
            else x for x in row] for row in file_content])

        return column_names , file_content


    def _check_parameters_fit_with_history(self, parameters: Parameters,
                                           column_names: list, history):
        """Checks that the parameters loaded in from the file of previous refinement steps are
         compatible with those already defined in the control object. If the parameters are the same
         but with different numbers (arbitrary), then this is changed to be consistent.

        Parameters
        ----------
        column_names: list
            A list of the columns names from a previous refinement file, excluding any step number
             column.
        history: list of lists
            A list, where each element is a subsequent list with parameter values and FoM value
             for each step in a previous refinement.

        Raises
        ----------
        ValueError
            If the number of parameters in the history is not consistent with the current set up.
            If the names of the parameters in the history are not the same as those in the
            current set up."""

        if history is not None:
            # using a reduced length for 'column_names' because it includes 'FoM'
            # and we want parameters only.
            if (len(column_names)-1) != len(parameters):
                raise ValueError(f'A history of {len(history.columns) -2}'\
                    ' is incompatible with the current setup.')

            split_param_list = [parameter.split(" ")[0] for parameter in parameters]
            split_column_list = [column.split(" ")[0] for column in column_names[1:]]
            if split_param_list != split_column_list:
                raise ValueError("The parameters in the minimizer history are not \
                                      the same as those specified for refining in the current\
                                      universe setup.")
            param_list = list(parameters)
            param_list.insert(0, 'Fom')
            self.column_names = param_list


    def get_parameters_old_values(self, parameters: Parameters, column_names: list, history):
        """Retrieves the last set of parameters from a file containing data of previous
        refinement steps.

        Parameters
        ----------
        column_names: list
            A list of the columns names for the past refinement data.
        history: list of lists
            A list, where each element is a subsequent list with parameter values and FoM value
             for each step in a previous refinement.

        Raises
        ----------
        Exception
            If the last parameter values can not be retrieved from history.

        Returns
        ----------

        dict (if there is a history file loaded:)
            dictionary of parameter values from the last step.

        None (if there is no history file loaded)
            None type.
        """
        if history:
            try:
                last_entry = history[-1]
                for param in parameters:
                    parameters[param].value = last_entry[column_names.index(param)]
            except Exception as err:
                raise Exception('Issue retrieving most recent parameter values \
                    from given results file.') from err
            return parameters
        return None

    def different_minimizer_compatibility(self, column_names, _history) -> None:
        """
        Checks that the refinement file has the correct set up to be used with the current minizer,
        and makes the necessary changes for this compatibility.

        Parameters
        ----------
        column_names: list
            A list of the columns names from a previous refinement file, excluding any step number
             column.
        history: list of lists
            A list, where each element is a subsequent list with parameter values and FoM value
             for each step in a previous refinement.

        Raises
        ----------
        Exception
            If changing _history for minimizer compatibility fails.

        """
        if _history and self.compatible is False:
            try:
                if 'Change state' in column_names and \
                    ('Accepted' not in _history[0] or 'Rejected' not in _history[0]):
                    for row in _history:
                        pos = column_names.index('Change state')
                        row.insert(pos,'Accepted')

                elif 'Change state' not in column_names and \
                    ('Accepted' in _history[0] or 'Rejected' in _history[0]):
                    for row in _history:
                        try:
                            row.remove('Accepted')
                        except ValueError:
                            row.remove('Rejected')

                self.compatible = True
            except Exception as err:
                raise Exception("Failed to make the data compatible with the different minimizers \
                            used between refinements.") from err
        self._history = _history
