"""A module for all minimizers which can be iterated to refine the potential
parameters"""

from abc import ABC, abstractmethod

import pandas
from mpi4py import MPI
import pandas as pd

from MDMC.MD import Parameters
from MDMC.common.decorators import repr_decorator

# pylint: disable=c-extension-no-member
# to avoid MPI warnings


@repr_decorator('comm', 'FoM', 'FoM_old',
                'parameters', 'parameters_old_values')
class Minimizer(ABC):

    """
    An abstract class with methods common to all minimizers

    Parameters
    ----------
    parameters : Parameters or list of Parameter
        A `list` of ``Parameter`` objects which will be fit

    Attributes
    ----------
    comm : mpi4py.MPI.Intracomm
        MPI Intracomm which has all of the specified processors
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

    def __init__(self, parameters):

        # Use all available processors, as provided by MPI.COMM_WORLD
        self.comm = MPI.COMM_WORLD

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
    def history(self):
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
    def history_columns(self) -> list[str]:
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
    def _check_parameters(parameters: Parameters):
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


    def present_result(self):
        """
        Calculates and returns the most appropriate output for the minimiser class
        e.g. minimum FOM and parameter values
        """
        raise NotImplementedError()

    def format_result_string(self,
                             last_parameters_found: tuple,
                             last_FoM_value: float,
                             lowest_FoM_parameters: tuple,
                             lowest_FoM_value: float) -> str:
        """
        Formats a string output for the results of the minimiser class

        Parameters
        ----------
        last_parameters_found: tuple
            A tuple-like object containing the parameters of the last point
            in the history

        last_FoM_value: float
            The FoM value of the last point in the history

        lowest_FoM_parameters: tuple
            A tuple-like object containing the parameters of the point with

        lowest_FoM_value: float
            The lowest FoM value of the history


        Returns
        -------
        str
            A string encompassing the output of the minimizer, in the following format:
            1. Whether or not the minimizer has converged
            2. The last parameters of the run
            3. The last FoM value of the run
            4. The optimal (lowest FoM) parameters
            5. The optimal (lowest) FoM value
        """

        if isinstance(last_parameters_found, tuple) and isinstance(lowest_FoM_parameters, tuple):
            if isinstance(last_FoM_value, float) and isinstance(lowest_FoM_value, float):
                has_converged = self.has_converged()
                converged_message = '\nThe refinement has converged.' if has_converged else "\nThe refinement has not converged."

                output_string = (f'{converged_message} \n \n'
                                 f'Last accepted point is: \n'
                                 f'{last_parameters_found} with a minimum '
                                 f'FoM of {last_FoM_value}. \n \n'
                                 f'Best point measured was: \n'
                                 f'{lowest_FoM_parameters} for a minimum FoM of '
                                 f'{lowest_FoM_value}.\n \n ')

                return output_string
            else:
                raise TypeError("The FoM values provided were not given as floats")
        else:
            raise TypeError("The parameters were not given as a tuple")
