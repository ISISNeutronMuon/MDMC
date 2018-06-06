"""Module for SQW class

AUTHOR :    Thomas Farmer        START DATE :    2018-6-6 13:24:27"""

import numpy as np
import uncertainties.unumpy as unp

from MDMC.src.trajectory_analysis.observables.exp_obs import \
    ExperimentalObservable

class DynamicStructureFactor(ExperimentalObservable):

    """
    A class for containing, calculating and reading a dynamic structure factor
    """

    def __init__(self):
        pass
