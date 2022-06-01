"""A module for all minimizers which can be iterated to refine the potential
parameters"""

from abc import ABC, abstractmethod
from mpi4py import MPI
import numpy as np
import pandas as pd

from MDMC.MD import Parameters
from MDMC.common.decorators import repr_decorator

# pylint: disable=c-extension-no-member
# to avoid MPI warnings


@repr_decorator('comm', 'FoM', 'FoM_old', 'distribution',
                'state_changed', 'parameters', 'parameters_old_values',
                'max_parameter_change')
class Minimizer(ABC):

    """
    An abstract class with methods common to all minimizers

    Parameters
    ----------
    parameters : Parameters or list of Parameter
        A `list` of ``Parameter`` objects which will be fit
    distribution : str, optional
        The distribution from which ``Parameter`` changes are selected

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
    state_changed : bool
        If the MMC algorithm resulted in the step being Accepted or Rejected
    max_parameter_change : float, optional
        Maximum factor by which a Parameter can change each step of the
        refinement. Defaults to `0.01`
    """

    DISTRIBUTION = {'uniform': np.random.uniform}

    def __init__(self, parameters, distribution='uniform',
                 max_parameter_change: float = 0.01):

        # Use all available processors, as provided by MPI.COMM_WORLD
        self.comm = MPI.COMM_WORLD

        # Parameters are only changed by rank 0 process, and so only rank 0
        # Minimizer needs a random distribution
        if self.comm.rank == 0:
            self.distribution = self.__class__.DISTRIBUTION[distribution]
        else:
            self.distribution = None

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
        self.max_parameter_change = max_parameter_change

        # Records if most recent step changed the state
        self.state_changed = None

    @abstractmethod
    def step(self):
        """
        Increments the minimization by a step
        """

        raise NotImplementedError

    @property
    def max_parameter_change(self):
        """
        Maximum factor by which a Parameter can change

        Returns
        -------
        float
            Maximum ``Parameter`` value change
        """

        return self._max_parameter_change

    @max_parameter_change.setter
    def max_parameter_change(self, value):

        self._max_parameter_change = value

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
    def history_columns(self):
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
    def change_state(self):
        """
        Stochastic determination of whether the state should change based on the
        FOM

        Returns
        -------
        bool
            `True` if the state should be changed
        """

        raise NotImplementedError

    @abstractmethod
    def change_parameters(self, parameters: Parameters):
        """
        Selects a new value for each ``Parameter`` from a distribution centered
        around the current value

        Parameters
        ----------
        parameters : Parameters
            All ``Parameter`` objects that are being refined
        """

        raise NotImplementedError

    def _calc_max_parameter_change(self):

        raise NotImplementedError

    def has_converged(self, conv_tol: float = 1e-5, min_steps: int = 2) -> bool:
        """
        Checks if the refinement process has converged on a stable solution.
        Specifically, it checks if the Figure of Merit and the parameters being refined have all
        changed less than the relative conversion tolerance (`conv_tol`) between the
        last two accepted refinement steps. It also allows specifying a minimum
        number of refinement steps (`min_steps`) that must have been accepted
        before checking for convergence.

        Parameters
        ----------
        conv_tol : float, optional
            The relative tolerance of the convergence check. Defaults to `1e-5`
        min_steps : int, optional
            The number of refinement steps with an accepted state change after which
            convergence is checked. If the number of accepted state changes is less than this,
            then the refinement is deemed as not converged.
            Defaults to `2`.

        Returns
        -------
        bool
            Whether or not the minimizer has converged.
        """

        # select the history of accepted state changes
        accepted_history = (self.history['Change state'] == 'Accepted')
        accepted_history = self.history[accepted_history]
        if len(accepted_history) >= min_steps:
            # drop 'Change state' column to select only parameters;
            # turn to np.array for easy slicing
            param_history = np.array(
                accepted_history.drop('Change state', axis=1))
            converged = np.allclose(
                param_history[-1], param_history[-2], rtol=conv_tol)
        else:
            converged = False

        return converged

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

    def write_history(self, filename):
        """
        Write the minimizer history to a csv file

        Parameters
        ----------
        filename : str
            The name of the output file
        """

        self.history.to_csv(filename)

    @abstractmethod
    def present_result(self):
        """
        Returns the most appropriate output for the minimiser class
        e.g. minimum FOM and parameter values
        """
        raise NotImplementedError
