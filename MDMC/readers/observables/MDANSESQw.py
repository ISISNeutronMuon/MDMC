"""Readers for dynamic data"""

import logging
import numpy as np

from MDMC.readers.observables.obs_reader import SQwReader

logger = logging.getLogger(__name__)

class MDANSESQw(SQwReader):
    """
    A class for reading SQw files from MDANSE

    The output from MDANSE analysis of trajectories is a .csv file with some lines
    of comments describing the dataset and columns/rows. The first rwo is the Q
    points of the dataset. The first column is the omega points for the dataset
    and the remaining points are the S(Q,w) at each corresponding Q and w.

    Attributes
    ----------
    file_variables : ndarray
        numpy array containing all the data
    """

    def __init__(self, file_name: str):
        super().__init__(file_name)
        self.file_variables = None

    def __enter__(self) -> None:
        """Open the files for variables and detector momenta"""
        # pylint: disable=consider-using-with
        # as this is an abstracted open method

        self.file_variables = np.loadtxt(self.file_name, delimiter=',')


    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """Does nothing since numpy closes the file after reading anyway"""
        pass

    def parse(self, **settings: dict) -> None:
        """
        Parse into SQw format

        E is the energy transfer (in meV)
        Q is wavevector transfer (in Ang^-1)
        """

        self.Q = self.file_variables[1:, 0]  # Entry ar [0,0] is always zero
        self.E = self.file_variables[0, 1:]
        self.SQw = self.file_variables[1:, 1:]
        self.SQw_err = np.sqrt(self.SQw)*0.01  # This is arbirtary and may need to be changed

        # Change and zero errors into inf so that error calculations can still be performed on them.
        if np.any(self.SQw_err <= 0.):
            self.SQw_err[np.where(self.SQw_err <= 0.)] = float('inf')
            msg = "We have set the error bar to infinity for any zero error values, this allows\
                us to calculate chi-squared but effectively ignores these points, this may not\
                be what you want to do, consider using a FoM which doesn't need errors if\
                this is an issue"
            logger.warning(msg)
