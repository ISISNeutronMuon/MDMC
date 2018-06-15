"""A module for Figure of Merits

AUTHOR :    Thomas Farmer        START DATE :    2018-6-15 14:15:58"""

import numpy as np

from abc import ABCMeta, abstractmethod

class FigureOfMerit:

    """
    Abstract class that defines methods common to all figures of merit
    """

    __metaclass__ = ABCMeta





    def check_dataset_properties(self):

        """
        Checks for required properties of all datasets

        This includes:
        - Two datasets exist
        - Exactly one dataset is an experimental dataset
        """

        raise NotImplementedError
