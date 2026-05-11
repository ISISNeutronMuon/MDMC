# MDMC is a package for the optimisation of classical potentials with experimental data
# Copyright (C) 2026 MDMC Developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Readers for dynamic data"""

import logging
from typing import Any

import numpy as np

from MDMC.common.units import SYSTEM, Unit
from MDMC.readers.observables.obs_reader import SQwReader

logger = logging.getLogger(__name__)

eV_in_Joules = 1.602176634 * 10 ** (-19)
mole = 6.02214076 * 10**23

conversion_to_meV = {
    "J": 6.2415091e21,
    "kJ": 6.2415091e24,
    "kcal": 2.6114474e25,
    "cal": 2.6114474e22,
    "kJ/mol": 6.2415091e24 / mole,
    "kcal/mol": 2.6114474e25 / mole,
    "J/mol": 6.2415091e21 / mole,
    "cal/mol": 2.6114474e22 / mole,
    "rad/ps": 0.6582231395941951,
    "rad/fs": 0.6582231395941951 * 1e3,
    "rad/ns": 0.6582231395941951 * 1e-3,
    "1/ps": 0.6582231395941951 * 2 * np.pi,
    "1/fs": 0.6582231395941951 * 1e3 * 2 * np.pi,
    "1/ns": 0.6582231395941951 * 1e-3 * 2 * np.pi,
    "meV": 1.0,
    "eV": 1e3,
    "keV": 1e6,
    "ueV": 1e-3,
}


class MDANSESQw(SQwReader):
    """
    A class for reading SQw files from MDANSE's trajectory analysis

    The output from MDANSE analysis of trajectories is a .csv file with some lines
    of comments describing the dataset and columns/rows, followed by an array
    of numbers. The first row and column of the array define the axes of the data,
    where the role and physical unit of each axis is described in the comment lines
    preceding the data. The [0,0] element of the array is always 0.0 and is not used,
    while all the remaining points are the S(Q,w) at each corresponding Q and w.

    Attributes
    ----------
    file_variables : ndarray
        numpy array containing all the data
    """

    def __init__(self, file_name: str):
        super().__init__(file_name)
        self.file_variables = None
        self.first_row = "Q"
        self.first_column = "E"
        self.q_unit = None
        self.e_unit = None
        self.transpose_data = True

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
        value = None
        # This loop will only read the header of the file,
        # and stop as soon as it reaches the data
        with open(self.file_name, "r", encoding="utf-8") as source:
            for line in source:
                tokens = line.split()
                if len(tokens) == 0:
                    continue
                if "#" in tokens[0]:
                    header.append(line)
                else:
                    break
        # This part will find the relevant part of the header
        # and extract the information about the axes.
        for line in header:
            if "1st" in line or "First" in line:
                axis_signature = line.split(":")[-1]
                variable = axis_signature.split()[0]
                unit = axis_signature.split()[1].strip("()")
                unit.replace("ang", "Ang")  # we need this since Unit cannot handle 'ang'
                if variable == "q":
                    value = "Q"
                    q_unit = Unit(unit)
                    try:
                        _ = q_unit.conversion_factor
                    except KeyError:
                        logger.warning("Unit %s not recognised, replaced with 1/Ang", str(unit))
                        q_unit = Unit("1") / SYSTEM["LENGTH"]
                    self.q_unit = q_unit
                elif variable == "omega":
                    value = "E"
                    try:
                        conversion_to_meV[unit]
                    except KeyError:
                        self.e_unit = "arb. u."
                    else:
                        self.e_unit = unit
                else:
                    raise ValueError(f"Unknown variable ({variable}).")
            if "row:" in line:
                self.first_row = value
            if "column:" in line:
                self.first_column = value
        if self.first_row != "Q":
            self.transpose_data = False

    def parse(self, **settings: Any) -> None:
        """
        Parse into SQw format, creates an error on SQw 1% of the value of SQw,
        since MDANSE does not yet output an error. This should be changed once
        it is possible to read an error.

        E is the energy transfer (in meV)
        Q is wavevector transfer (in Ang^-1)
        """
        self.parse_header()

        if self.first_row == "Q" and self.first_column == "E":
            self.Q = self.file_variables[0, 1:]  # Entry at [0,0] is always zero
            self.E = self.file_variables[1:, 0]
        elif self.first_row == "E" and self.first_column == "Q":
            self.E = self.file_variables[0, 1:]
            self.Q = self.file_variables[1:, 0]
        self.Q *= self.q_unit.conversion_factor
        self.E *= conversion_to_meV[self.e_unit]
        if self.transpose_data:
            self.SQw = self.file_variables[1:, 1:].T
        else:
            self.SQw = self.file_variables[1:, 1:]
        self.SQw_err = self.SQw * 0.01  # TODO: When MDANSE outputs an error, read it in
        if np.any(self.SQw <= 0.0):
            self.SQw[np.where(self.SQw <= 0.0)] = 0.0
        # Change and zero errors into inf so that error calculations can still be performed on them.
        if np.any(self.SQw_err <= 0.0):
            self.SQw_err[np.where(self.SQw_err <= 0.0)] = float("inf")
            msg = "We have set the error bar to infinity for any zero error values, this allows\
                us to calculate chi-squared but effectively ignores these points, this may not\
                be what you want to do, consider using a FoM which doesn't need errors if\
                this is an issue"
            logger.warning(msg)
