"""A module for all minimizers which can be iterated to refine the potential
parameters

AUTHOR :    Thomas Farmer        START DATE :    2018-4-26 10:51:42"""

from abc import ABCMeta, abstractmethod, abstractproperty
import random
from copy import deepcopy

import numpy as np

class Minimizer:

    """
    An abstract class with methods common to all minimizers
    """

    __metaclass__ = ABCMeta

    def __init__(self, MC_norm, config_reset):

        """
        Arguments:
        MC_norm - Normalization parameter for MC which determines the
        accept/reject ratio
        config_reset - Boolean which determines whether or not the MD
        configuration is stored.  If True, the MD configuration will always be
        reset to the configuration for the last accepted parameter set.
        """

        self.FoM_old = float('inf')
        self.FoM = None
        self.fit_params_old = None
        self.fit_params = None
        self.MC_norm = MC_norm
        self.config_reset = config_reset

    @abstractmethod
    def step(self):

        raise NotImplementedError

    @property
    def fit_params_old(self):

        return self._fit_params_old

    @fit_params_old.setter
    def fit_params_old(self, params):

        self._fit_params_old = deepcopy(params)

    @abstractproperty
    def max_param_change(self):

        raise NotImplementedError

    @abstractmethod
    def change_state(self):

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

    def change_parameters(self, fit_params):

        """
        Arguments:
        fit_params - References to all potential parameters that will be refined
        """

        for param in fit_params:
            self._change_parameter(param)

    def _calc_max_param_change(self):

        raise NotImplementedError

    def has_converged(self):

        raise NotImplementedError


class MMC(Minimizer):

    """
    Minimizer employing the Metropolis algorithm
    """

    def step(self, FoM):

        self.FoM = FoM

        if self.change_state():
            self.FoM_old = self.FoM
            # TODO: FINISH!

    def change_state(self):

        prob = min(1, np.exp(self.FoM_old - self.FoM) / self.MC_norm)
        return True if prob > random.random() else False

    def _change_parameter(self, parameter):

        parameter.value += parameter.value * \
            random.uniform(-self.max_param_change,self.max_param_change)
