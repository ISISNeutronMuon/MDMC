"""A module for Figure of Merits

AUTHOR :    Thomas Farmer        START DATE :    2018-6-15 14:15:58"""

from abc import ABCMeta, abstractmethod

import numpy as np

class FigureOfMeritCalculator:

    """
    Abstract class that defines methods common to all figure of merit
    calculators
    """

    __metaclass__ = ABCMeta

    def __init__(self, obs_pairs):

        """
        Arguments:
        obs_pairs - One or more ObservablePairs
        """

        self.obs_pairs = list(obs_pairs)
        self.value = np.sum([self.calculate_FoM(obs_pair)
                             for obs_pair in self.obs_pairs])

    @abstractmethod
    def calculate_single_FoM(self, obs_pair):

        """
        Performs the FoM calculation specific to each FoM
        """

        raise NotImplementedError


class StandardFoMCalculator(FigureOfMeritCalculator):

    """
    Calculates the error normalised square difference, with an optional
    weighting
    """

    def calculate_single_FoM(self, obs_pair):

        return np.sum(obs_pair.calculate_diffence() ** 2
                      * obs_pair.weight / obs_pair.calculate_errors())


class ObservablePair(object):

    """
    Contains a pair of observables for calculating the FoM

    Checks the validity of observables
    """

    def __init__(self, exp_obs, MD_obs, weight):

        """
        Arguments:
        exp_obs - an Observable with an origin 'experiment'
        MD_obs - an Observable with an origin 'MD'
        weight - a float with the relative weight of this pair on the total FoM
        """

        self.exp_obs = exp_obs
        self.MD_obs = MD_obs
        self.weight = weight

    @property
    def exp_obs(self):

        return self._exp_obs

    @exp_obs.setter
    def exp_obs(self, exp_obs):

        self.validate_obs(exp_obs, 'experiment')
        self._exp_obs = exp_obs

    @property
    def MD_obs(self):

        return self._MD_obs

    @MD_obs.setter
    def MD_obs(self, MD_obs):

        self.validate_obs(MD_obs, 'MD')
        self._MD_obs = MD_obs

    def validate_obs(self, obs, origin):

        """
        Performs all applicable checks to test the validity of an observable
        """

        raise NotImplementedError

    def check_types(self):

        """
        Checks that observables are of the same type
        """

        raise NotImplementedError

    def check_indep_var(self):

        """
        Checks that observables have the same independent variables and that are
        finite
        """

        raise NotImplementedError

    def check_dep_var(self):

        """
        Checks that an observable has dependent variable data that are finite
        """

        raise NotImplementedError

    def check_errors(self):

        """
        Checks that an observable has errors on the dependent variable and that
        these are floats and not NaN
        """

        raise NotImplementedError

    def check_origin(self, origin):

        """
        Checks that the origin ('experiment' or 'MD') is correct
        """

        raise NotImplementedError

    def calculate_diffence(self):

        """
        Returns:
        The difference between the dependent variables
        """

        raise NotImplementedError

    def calculate_errors(self):

        """
        Returns:
        The combination of the errors in quadrature
        """

        raise NotImplementedError
