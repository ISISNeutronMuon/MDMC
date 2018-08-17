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
        """


        # First MC step always changes state
        self.FoM_old = float('inf')
        self.FoM = None

        params = np.array(list(params))
        self._check_parameters(params)
        self.params_old_values = np.array([param.value for param in params])
        self.params = self.change_parameters(params)
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

        if self.change_state():
            self.FoM_old = self.FoM
            # TODO: FINISH!

    def change_state(self):

        prob = min(1, np.exp(self.FoM_old - self.FoM) / self.MC_norm)
        return True if prob > random.random() else False

    def _change_parameter(self, parameter):

        parameter.value += parameter.value * \
            random.uniform(-self.max_param_change,self.max_param_change)
