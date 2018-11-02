"""A module for all minimizers which can be iterated to refine the potential
parameters

AUTHOR :    Thomas Farmer        START DATE :    2018-4-26 10:51:42"""


from abc import ABCMeta, abstractmethod

import numpy as np


class Minimizer:

    """
    An abstract class with methods common to all minimizers
    """

    __metaclass__ = ABCMeta

    DISTRIBUTION = {'normal':np.random.normal,
                    'uniform':np.random.uniform}

    def __init__(self, MC_norm, params, config_reset=False,
                 distribution='normal'):

        """
        Arguments:
        MC_norm - Normalization parameter for MC which determines the
        accept/reject ratio
        params - a list of MD parameters which will be fit
        config_reset - Boolean which determines whether or not the MD
        configuration is stored.  If True, the MD configuration will always be
        reset to the configuration for the last accepted parameter set.
        distribution - the distribution from which parameter changes are
        selected
        """

        self.distribution = self.__class__.DISTRIBUTION[distribution]

        # First MC step always changes state
        self.FoM_old = float('inf')
        self.FoM = None

        # History of minimization
        self.history = []

        params = np.array(list(params))
        self._check_parameters(params)
        self.params_old_values = None
        self.params = params
        self.MC_norm = MC_norm
        self.config_reset = config_reset

    @abstractmethod
    def step(self):

        """
        Increments the minimization by a step
        """

        raise NotImplementedError

    @property
    def max_param_change(self):

        """
        Maximum factor by which a parameter can change
        """

        return 0.01

    @abstractmethod
    def change_state(self):

        """
        Stochastic determination of whether the state should change based on the
        FOM
        """

        raise NotImplementedError

    @abstractmethod
    def change_parameters(self, params):

        """
        Selects a new value for each parameter from a distribution centered
        around the current value

        Arguments:
        params - References to all potential parameters that will be refined
        """

        raise NotImplementedError

    def _calc_max_param_change(self):

        raise NotImplementedError

    def has_converged(self):

        return False

    def _check_parameters(self, params):

        """
        Checks the validity of the parameters on input

        Raises:
        ValueError when any parameter has fixed = True
        """

        for param in params:
            if param.fixed == True:
                raise ValueError('Parameter {0} is fixed'.format(param.name))


class MMC(Minimizer):

    """
    Minimizer employing the Metropolis-Hastings algorithm
    """

    def step(self, FoM):

        self.FoM = FoM
        values = np.array([p.value for p in self.params])
        print '\n' + 'New FoM'
        print self.FoM
        print 'Old FoM'
        print self.FoM_old
        print values
        history = [self.FoM, values]

        if self.change_state():
            print 'Accepted'
            history.append('Accepted')
            self.FoM_old = self.FoM
            self.params_old_values = np.array([param.value
                                               for param in self.params])

        else:
            print 'Rejected'
            history.append('Rejected')
            self.FoM = self.FoM_old
            self.reset_params()

        self.history.append(history)
        self.change_parameters(self.params)

    def change_state(self):

        prob = min(1, np.exp((self.FoM_old - self.FoM) / self.MC_norm))
        return True if prob > np.random.random() else False

    def change_parameters(self, params):

        # Faster to generate all random numbers at once
        changes = self.distribution(-self.max_param_change,
                                    self.max_param_change,
                                    len(params))
        for i, param in enumerate(params):
            param.value += param.value * changes[i]

    def reset_params(self):

        for i, param in enumerate(self.params):
            param.value = self.params_old_values[i]
