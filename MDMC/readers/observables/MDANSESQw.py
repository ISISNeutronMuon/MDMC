"""Readers for dynamic data"""

import logging
import numpy as np

from MDMC.readers.observables.obs_reader import SQwReader
from MDMC.common.units import SYSTEM, Unit

logger = logging.getLogger(__name__)

class MDANSESQw(SQwReader):
    """
    A class for reading SQw files from MDANSE's trajectory analysis

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
        self.first_row = 'Q'
        self.first_column = 'E'
        self.q_unit = None
        self.e_unit = None

    def __enter__(self) -> None:
        """Open the files for variables and detector momenta"""
        # pylint: disable=consider-using-with
        # as this is an abstracted open method
        self.file_variables = np.loadtxt(self.file_name)


    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """Does nothing since numpy closes the file after reading anyway"""
        # pylint: disable=unnecessary-pass
        pass

    def parse_header(self):
        """Reads the header part of the file to get the labels of
        the data axes.
        """
        header = []
        # This loop will only read the header of the file,
        # and stop as soon as it reaches the data
        with open(self.file_name, 'r') as source:
            for line in source:
                tokens = line.split()
                if len(tokens) == 0:
                    continue
                elif '#' in tokens[0]:
                    header.append(line)
                else:
                    break
        # This part will find the relevant part of the header
        # and extract the information about the axes.
        for line in header:
            if '1st' in line or 'First' in line:
                axis_signature = line.split(':')[-1]
                variable = axis_signature.split()[0]
                unit = axis_signature.split()[1].strip("()")
                if variable == 'q':
                    value = 'Q'
                    q_unit = Unit(unit)
                    try:
                        q_unit.conversion_factor
                    except KeyError:
                        q_unit = 1/SYSTEM["LENGTH"]
                    self.q_unit = q_unit
                elif variable == 'omega':
                    value = 'E'
                    e_unit = Unit(unit)
                    try:
                        e_unit.conversion_factor
                    except KeyError:
                        e_unit = SYSTEM["ENERGY_TRANSFER"]
                    self.e_unit = e_unit
            if "row:" in line:
                self.first_row = value
            if "column:" in line:
                self.first_column = value

    def parse(self, **settings: dict) -> None:
        """
        Parse into SQw format, creates an error on SQw 1% of the value of SQw,
        since MDANSE does not yet output an error. This should be changed once
        it is possible to read an error.

        E is the energy transfer (in meV)
        Q is wavevector transfer (in Ang^-1)
        """
        self.parse_header()

        if self.first_row == 'Q' and self.first_column == 'E':
            self.Q = self.file_variables[0, 1:] # Entry at [0,0] is always zero
            self.E = self.file_variables[1:, 0]
        elif self.first_row == 'E' and self.first_column == 'Q':
            self.E = self.file_variables[0, 1:]
            self.Q = self.file_variables[1:, 0]
        self.Q *= self.q_unit.conversion_factor
        self.E *= self.e_unit.conversion_factor
        self.SQw = self.file_variables[1:, 1:].T
        self.SQw_err = self.SQw*0.01  # TODO: When MDANSE outputs an error, read it in
        if np.any(self.SQw <= 0.):
            self.SQw[np.where(self.SQw <= 0.)] = 0.0
        # Change and zero errors into inf so that error calculations can still be performed on them.
        if np.any(self.SQw_err <= 0.):
            self.SQw_err[np.where(self.SQw_err <= 0.)] = float('inf')
            msg = "We have set the error bar to infinity for any zero error values, this allows\
                us to calculate chi-squared but effectively ignores these points, this may not\
                be what you want to do, consider using a FoM which doesn't need errors if\
                this is an issue"
            logger.warning(msg)
