"""A module for Figure of Merits

AUTHOR :    Thomas Farmer        START DATE :    2018-6-15 14:15:58"""

from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np

class FigureOfMeritCalculator:

    """
    Abstract class that defines methods common to all figure of merit
    calculators
    """

    __metaclass__ = ABCMeta

    def calculate_all_FoM(self, obs_pairs):

        """
        Arguments:
        obs_pair - an ObservablePair
        """

        FoMs = []
        for obs_pair in obs_pairs:
            FoMs.append(self.calculate_FoM(obs_pair))

        return sum(FoMs)

    @abstractmethod
    def calculate_FoM(self, obs_pair):

        """
        Performs the FoM calculation specific to each FoM
        """

        raise NotImplementedError


class StandardFoMCalculator(FigureOfMeritCalculator):

    """
    Calculates the error normalised square difference, with an optional
    weighting
    """

    def calculate_FoM(self, obs_pair):

        return np.sum(obs_pair.calculate_diffence() ** 2
                      * obs_pair.weight / obs_pair.calculate_errors())

        """

        """

        raise NotImplementedError







        raise NotImplementedError
