"""Module for coherent SQw class

AUTHOR :    Thomas Farmer        START DATE :    20/07/2018, 16:28:02"""

import numpy as np

from MDMC.src.common.mathematics import correlation
from MDMC.src.trajectory_analysis.observables.SQw import AbstractSQw


class SQwCoherent(AbstractSQw):

    """
    A class for containing, calculating and reading the coherent dynamic
    structure factor
    """

    def _calculate_FQt_single_Q(self, Q_vector):

        raise NotImplementedError

    def _calculate_rho(self, positions, Q_vector):

        raise NotImplementedError
