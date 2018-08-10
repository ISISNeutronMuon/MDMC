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

    def calculate_all_FoM(self, obs_pair):

        """
        Accepts a list of dictionaries of data. Each
        dictionary contains exp_data, MD_data, exp_err, and optionally contains
        weight.
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

    @abstractmethod
    def check_data_properties(self, data_pairs):

        """
        Checks for required properties of all datasets

        This includes:
        - At least two datasets exist
        - Exactly one dataset in each pair is an experimental dataset
        - Identical dimensions for each pair
        """

        raise NotImplementedError


class StandardFoMCalculator(FigureOfMeritCalculator):

    """
    Calculates the error normalised square difference, with an optional
    weighting
    """

    def calculate_FoM(self, data_pair):

        return np.sum((data_pair['exp_data'] - data_pair['MD_data']) ^ 2 \
            * data_pair.get('weight', 1) / data_pair['err_data'])

    # TODO: Implement
    def check_data_properties(self):

        raise NotImplementedError
