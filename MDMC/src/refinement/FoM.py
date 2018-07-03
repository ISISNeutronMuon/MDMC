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

    @abstractmethod
    def calculate_FOM(self):

        raise NotImplementedError

    @abstractmethod
    def check_dataset_properties(self):

        """
        Checks for required properties of all datasets

        This includes:
        - Two datasets exist
        - Exactly one dataset is an experimental dataset
        - Identical dimensions
        """

        raise NotImplementedError


class MultipleFOMCalculator(FigureOfMeritCalculator):

    """
    Enables a single FOM to be calculated from multiple FOM (i.e. multiple
    experimental datasets)

    Each individual FOM has its own scaling, which defaults to unity.
    """

    def __init__(self, *data):

        """
        data_pair - a dictionary containing exp_data and MD_data
        """


class StandardFOMCalculator(FigureOfMeritCalculator):

    """
    Calculates the error normalised square difference
    """

    def __init__(self, data):

        """
        data - a dictionary containing exp_data, MD_data, err_data and scale,
        which is a scale factor for normalising the MD data to the exp_data
        """

        self.exp_data = data.get("exp_data")
        self.MD_data = data.get("MD_data")
        self.err_data = data.get("err_data")
        self.scale = data.get("scale")
        self.check_dataset_properties()

    def calculate_FOM(self):

        return np.sum((self.exp_data - self.scale * self.MD_data) ^ 2 \
            / self.err_data)

    # TODO: Implement
    def check_dataset_properties(self):

        pass
