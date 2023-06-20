"""A module for all minimizers which can be iterated to refine the potential
parameters"""
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

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

    def __init__(self, control: 'Control', parameters: Parameters):
        self.control = control

        # First MC step always changes state
        self.FoM_old = float('inf')
        self.FoM = None

        # History of minimization
        self._history = []

        if isinstance(parameters, list):
            parameters = Parameters(parameters)

        self._check_parameters(parameters)
        self.parameters_old_values = None
        self.parameters = parameters

        # Records if most recent step changed the state
        self.state_changed = None

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

        return pd.DataFrame(self._history, columns=self.history_columns)

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
