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
        self.value = None

    def calculate(self):

        """
        Calculates the FoM value by calculating the FoM for every observable
        pair

        Returns:
        Non-negative float
        """

        self.value = np.sum([self.calculate_single_FoM(obs_pair)
                             for obs_pair in self.obs_pairs])
        assert self.value >= 0.
        return self.value

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

        return np.sum(obs_pair.calculate_difference() ** 2
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

    @property
    def weight(self):

        return self._weight

    @weight.setter
    def weight(self, weight):

        try:
            weight = float(weight)
        except ValueError:
            raise TypeError('weight must be a float')
        self.validate_weight(weight)
        self._weight = weight

    def validate_obs(self, obs, origin):

        """
        Performs checks to test the validity of an observable

        Arguments:
        obs - an osbervable
        origin - a string specifying the origin of the observable ('experiment'
        or 'MD')
        """

        # Check origin is correct
        assert obs.origin == origin, ('The observable does not have the correct'
                                      ' origin')

        try:
            if obs.origin == 'MD':
                other_obs = self.exp_obs
            else:
                other_obs = self.MD_obs
        except AttributeError:
            other_obs = None

        # Check independent variables are identical, check dependent variables
        # have the same shapes, check errors have the same shapes, check
        # observables have the same type
        if other_obs:
            indep_e_mess = 'Independent variables must be identical'
            assert (obs.independent_variables.keys() ==
                    other_obs.independent_variables.keys()), indep_e_mess
            for k in obs.independent_variables.keys():
                assert np.all(obs.independent_variables[k] ==
                              other_obs.independent_variables[k]), indep_e_mess

            dep_e_mess = 'Dependent variables must have the same shape'
            assert (obs.dependent_variables.keys() ==
                    other_obs.dependent_variables.keys()), dep_e_mess
            for k in obs.dependent_variables:
                assert (np.shape(obs.dependent_variables[k]) ==
                        np.shape(other_obs.dependent_variables[k])), dep_e_mess

            err_e_mess = 'Errors must have the same shape'
            assert obs.errors.keys() == other_obs.errors.keys(), err_e_mess
            for k in obs.errors:
                assert (np.shape(obs.errors[k]) ==
                        np.shape(other_obs.errors[k])), err_e_mess

            assert isinstance(obs, type(other_obs)), ('Observables are not of'
                                                      ' the same type')

    def validate_weight(self, weight):

        """
        Performs checks to test the validity of the weight

        Arguments:
        weight - the weight attribute
        """

        assert weight > 0. and weight != np.float('inf'), ('Weight must be a '
                                                           'finite non-negative'
                                                           ' float')


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

    def calculate_difference(self):

        """
        Assumes a single dependent variable for each observable

        Returns:
        The absolute difference between the dependent variables
        """

        diff = (np.array(self.exp_obs.dependent_variables.values())
                - np.array(self.MD_obs.dependent_variables.values()))

        return diff

    def calculate_errors(self):

        """
        Assumes a single dependent variable error for each observable

        Returns:
        The combination of the errors in quadrature
        """

        errors = (np.array(self.exp_obs.errors.values()) ** 2
                  + np.array(self.MD_obs.errors.values()) ** 2) ** 0.5

        return errors
