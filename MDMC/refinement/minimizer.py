"""A module for all minimizers which can be iterated to refine the potential
parameters"""


from abc import ABC, abstractmethod

from mpi4py import MPI
import numpy as np
import pandas as pd

from MDMC.common.decorators import repr_decorator


@repr_decorator('comm', 'FoM', 'FoM_old', 'MC_norm', 'distribution',
                'state_changed', 'params', 'params_old_values')
class Minimizer(ABC):

    """
    An abstract class with methods common to all minimizers

    Parameters
    ----------
    MC_norm : float
        Normalization parameter for MC which determines the accept/reject ratio
    params : list
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
    params : list
        A `list` of ``Parameter`` objects being fitted
    params_old_values : list
        A `list` of the values of the ``Parameter`` objects from the previous
        minimizer step
    state_changed : bool
        If the MMC algorithm resulted in the step being Accepted or Rejected
    """

    DISTRIBUTION = {'uniform':np.random.uniform}

    def __init__(self, MC_norm, params, distribution='uniform'):

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

        params = np.array(list(params))
        self._check_parameters(params)
        self.params_old_values = None
        self.params = params
        self.MC_norm = MC_norm

        # Records if most recent step changed the state
        self.state_changed = None

    @abstractmethod
    def step(self):

        """
        Increments the minimization by a step
        """

        raise NotImplementedError

    @property
    def max_param_change(self):

        """
        Maximum factor by which a Parameter can change

        Returns
        -------
        float
            Maximum ``Parameter`` value change
        """

        return 0.01

    @property
    def history(self):

        """

        """

        return pd.DataFrame(self._history, columns=self.history_columns)

    @property
    @abstractmethod
    def history_columns(self):

        """

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
    def change_parameters(self, params):

        """
        Selects a new value for each ``Parameter`` from a distribution centered
        around the current value

        Parameters
        ----------
        params : list
            All ``Parameter`` objects that are being refined
        """

        raise NotImplementedError

    def _calc_max_param_change(self):

        raise NotImplementedError

    def has_converged(self):

        return False

    def _check_parameters(self, params):

        """
        Checks the validity of the parameters on input

        Parameters
        ----------
        params : list
            All ``Parameter`` objects to validate

        Raises
        ------
        ValueError
            If any ``Parameter`` is fixed
        """

        for param in params:
            if param.fixed == True:
                raise ValueError('Parameter {0} is fixed'.format(param.name))

    def output_history(self, filename):

        self.history.to_csv(filename)


class MMC(Minimizer):

    """
    ``Minimizer`` employing the Metropolis-Hastings algorithm
    """

    @property
    def history_columns(self):

        return ['FoM', 'Old FoM', 'Change state'] + [p.name for p
                                                     in self.params]

    def step(self, FoM):

        """
        Increments the minimization by a step
        """

        self.FoM = FoM
        values = np.array([p.value for p in self.params])
        history = [self.FoM]

        if self.change_state():
            history.append('Accepted')
            self.FoM_old = self.FoM
            self.params_old_values = values
            self.state_changed = True

        else:
            history.append('Rejected')
            self.FoM = self.FoM_old
            self.reset_params()
            self.state_changed = False

        history.extend(values)
        self._history.append(history)
        self.change_parameters(self.params)

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

    def change_parameters(self, params):

        """
        Selects a new value for each parameter from a distribution centered
        around the current value

        Parameters
        ----------
        params : list
            All ``Parameter`` objects that are being refined
        """

        # Only calculate magnitude of parameter changes on rank 0 process, so
        # that each process ends up with same parameters
        if self.comm.rank == 0:
            # Faster to generate all random numbers at once
            changes = self.distribution(-self.max_param_change,
                                        self.max_param_change,
                                        len(params))
        else:
            changes = None
        # Broadcast parameters changes to all processes
        changes = self.comm.bcast(changes, root=0)
        # Change parameters by same amount on all processes
        for i, param in enumerate(params):
            param.value += param.value * changes[i]

    def reset_params(self):

        """
        Resets the ``Parameter`` values to the values from the previous MMC step
        """

        for i, param in enumerate(self.params):
            param.value = self.params_old_values[i]
