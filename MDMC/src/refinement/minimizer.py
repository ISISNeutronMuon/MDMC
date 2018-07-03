"""A module for all minimizers which can be iterated to refine the potential
parameters

AUTHOR :    Thomas Farmer        START DATE :    2018-4-26 10:51:42"""

from abc import ABCMeta, abstractmethod, abstractproperty

import random
import numpy as np

class Minimizer:

    """
    An abstract class with methods common to all minimizers
    """

    __metaclass__ = ABCMeta

    def __init__(self, MC_norm):

        """
        Arguments:
        fit_params - References to all potential parameters that will be refined
        """

        self.FoM_old = float('inf')
        self.FoM_new = None
        self.MC_norm = MC_norm

    def step(self, fit_params):

        """
        Iterates the minimization by a single step

        The following occurs with each step:
        - Accept/Reject new state
        - Change potential parameters
        """

        if self._change_state():
            self.FoM_old = self.FoM_new
            self._change_parameter(fit_params)

    @abstractproperty
    def max_param_change(self):

        raise NotImplementedError

    @abstractmethod
    def _change_state(self):

        """
        Stochastic determination of whether the state should change based on the
        FOM
        """

        raise NotImplementedError

    @abstractmethod
    def _change_parameter(self, parameter):

        """
        Selects a new value for the parameter from a distribution centered
        around the current value
        """

        raise NotImplementedError

    def _calc_max_param_change(self):

        raise NotImplementedError


class MMC(Minimizer):

    """
    Minimizer employing the Metropolis algorithm
    """

    def _change_state(self):

        prob = min(1, np.exp(self.FoM_old - self.FoM_new) / self.MC_norm)
        return True if prob > random.random() else False

    def _change_parameter(self, parameter):

        parameter.value += parameter.value * \
            random.uniform(-self.max_param_change,self.max_param_change)
