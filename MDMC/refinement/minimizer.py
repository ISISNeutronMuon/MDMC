"""A module for all minimizers which can be iterated to refine the potential
parameters"""


from abc import ABC, abstractmethod

from mpi4py import MPI
import numpy as np
import pandas as pd

from MDMC.common.decorators import repr_decorator


@repr_decorator('comm', 'FoM', 'FoM_old', 'MC_norm', 'distribution',
                'state_changed', 'parameters', 'parameters_old_values',
                'max_parameter_change')
class Minimizer(ABC):

    """
    An abstract class with methods common to all minimizers

    Parameters
    ----------
    MC_norm : float
        Normalization parameter for MC which determines the accept/reject ratio
    parameters : list
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
    parameters : list
        A `list` of ``Parameter`` objects being fitted
    parameters_old_values : list
        A `list` of the values of the ``Parameter`` objects from the previous
        minimizer step
    state_changed : bool
        If the MMC algorithm resulted in the step being Accepted or Rejected
    max_parameter_change : float, optional
        Maximum factor by which a Parameter can change each step of the
        refinement. Defaults to `0.01`
    """

    DISTRIBUTION = {'uniform':np.random.uniform}

    def __init__(self, MC_norm, parameters, distribution='uniform',
                 max_parameter_change: float=0.01):

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

        parameters = np.array(list(parameters))
        self._check_parameters(parameters)
        self.parameters_old_values = None
        self.parameters = parameters
        self.MC_norm = MC_norm
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
    def change_parameters(self, parameters):

        """
        Selects a new value for each ``Parameter`` from a distribution centered
        around the current value

        Parameters
        ----------
        parameters : list
            All ``Parameter`` objects that are being refined
        """

        raise NotImplementedError

    def _calc_max_parameter_change(self):

        raise NotImplementedError

    def has_converged(self, conv_tol: float=1e-3, min_steps: int=2) -> bool:

        """
        Checks if the refinement process has converged on a stable solution. Specifically, it checks if
        the Figure of Merit and the parameters being refined have all changed less than the relative conversion
        tolerance (`conv_tol`) between the last two accepted refinement steps. It also allows specifying a minimum
        number of refinement steps (`min_steps`) that must have been accepted before checking for convergence.

        Parameters
        ----------
        conv_tol : float, optional
            The relative tolerance of the convergence check. Defaults to `1e-3`
        min_steps : int, optional
            The number of refinement steps with an accepted state change after which convergence is checked. If the
            number of accepted state changes is less than this number then the refinement is deemed as not converged.
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
            # drop 'Change state' column to select only parameters; turn to np.array for easy slicing
            param_history = np.array(accepted_history.drop('Change state', axis=1))
            converged = np.allclose(param_history[-1], param_history[-2], rtol=conv_tol)
        else:
            converged = False

        return converged

    def _check_parameters(self, parameters):

        """
        Checks the validity of the parameters on input

        Parameters
        ----------
        parameters : list
            All ``Parameter`` objects to validate

        Raises
        ------
        ValueError
            If any ``Parameter`` is fixed
        """

        for parameter in parameters:
            if parameter.fixed is True:
                raise ValueError('Parameter {0} is fixed'.format(parameter.name))

    def write_history(self, filename):

        """
        Write the minimizer history to a csv file

        Parameters
        ----------
        filename : str
            The name of the output file
        """

        self.history.to_csv(filename)


class MMC(Minimizer):

    """
    ``Minimizer`` employing the Metropolis-Hastings algorithm
    """

    @property
    def history_columns(self):

        return ['FoM', 'Change state'] + [p.name for p in self.parameters]

    def step(self, FoM):

        """
        Increments the minimization by a step
        """

        self.FoM = FoM
        values = np.array([p.value for p in self.parameters])
        history = [self.FoM]

        if self.change_state():
            history.append('Accepted')
            self.FoM_old = self.FoM
            self.parameters_old_values = values
            self.state_changed = True

        else:
            history.append('Rejected')
            self.FoM = self.FoM_old
            self.reset_parameters()
            self.state_changed = False

        history.extend(values)
        self._history.append(history)
        self.change_parameters(self.parameters)

    def change_state(self):

        """
        Stochastic determination of whether the state should change based on the
        FoM

        Returns
        -------
        bool
            `True` if the state should be change
        """

        # Only determine if state will be changed on rank 0 process
        if self.comm.rank == 0:
            prob = min(1, np.exp((self.FoM_old - self.FoM) / self.MC_norm))
            change_state = True if prob > np.random.random() else False
        else:
            change_state = None
        # Broadcast to all processes whether or not the state will be changed
        change_state = self.comm.bcast(change_state, root=0)

        return change_state

    def change_parameters(self, parameters):

        """
        Selects a new value for each parameter from a distribution centered
        around the current value

        Parameters
        ----------
        parameters : list
            All ``Parameter`` objects that are being refined
        """

        # Only calculate magnitude of parameter changes on rank 0 process, so
        # that each process ends up with same parameters
        if self.comm.rank == 0:
            # Faster to generate all random numbers at once
            changes = self.distribution(-self.max_parameter_change,
                                        self.max_parameter_change,
                                        len(parameters))
        else:
            changes = None
        # Broadcast parameters changes to all processes
        changes = self.comm.bcast(changes, root=0)
        # Change parameters by same amount on all processes
        for i, parameter in enumerate(parameters):
            parameter.value += parameter.value * changes[i]

    def reset_parameters(self):

        """
        Resets the ``Parameter`` values to the values from the previous MMC step
        """

        for i, parameter in enumerate(self.parameters):
            parameter.value = self.parameters_old_values[i]
