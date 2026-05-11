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
from collections.abc import Iterable
from typing import IO, Any

import numpy as np

from MDMC.readers.observables.obs_reader import SQwReader

logger = logging.getLogger(__name__)

class LAMPSQw(SQwReader):

    """
    A class for reading SQw files from LAMP

    LAMP's ascii output uses three files: 1 for independent variables and
    parameters (..._LAMP), another for dependent variables
    (..._LAMPascii), and a third for the errors in the dependent variables
    (...LAMPascii_e)

    Attributes
    ----------
    file_indep : file
        File containing the independent variables
    file_dep : file
        File containing the dependent variables
    file_dep_err: file
        File containing the errors on the dependent variables
    """

    def __init__(self, file_name: str):
        super().__init__(file_name)
        self._Y_dim = None
        self._X_dim = None
        self.file_dep_err = None
        self.file_dep = None
        self.file_indep = None

    def __enter__(self) -> None:
        """
        Open the files for independent variables, dependent variables and errors
        on the dependent variables
        """
        # pylint: disable=consider-using-with
        # as this is an abstracted open method

        self.file_indep = open(self.file_name, encoding='UTF-8')
        self.file_dep = open(self.file_name + 'ascii', encoding='UTF-8')
        self.file_dep_err = open(self.file_name + 'ascii_e', encoding='UTF-8')

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """Closes all three files after parsing"""

        self.file_indep.close()
        self.file_dep.close()
        self.file_dep_err.close()

    def parse(self, **settings: Any) -> None:
        """
        Parse into SQw format

        E is the energy transfer (in meV)
        Q is wavevector transfer (in Ang^-1)
        """

        self.E, self.Q = self.parse_indep_var(self.file_indep)
        self.SQw = self.parse_dep_var(self.file_dep)
        self.SQw_err = self.parse_dep_var(self.file_dep_err)

        # LAMP sets errors -1 if the corresponding datum is 0.  Change these to
        # inf so that error calculations can still be performed on them but
        # result in inf.
        if np.any(self.SQw_err <= 0.):
            self.SQw_err[np.where(self.SQw_err <= 0.)] = float('inf')
            msg = "We have set the error bar to infinity for any zero error values, this allows\
                us to calculate chi-squared but effectively ignores these points, this may not\
                be what you want to do, consider using a FoM which doesn't need errors if\
                this is an issue"
            logger.warning(msg)

    def parse_indep_var(self, file: IO) -> tuple[np.ndarray, np.ndarray]:
        """
        Parses the independent variables

        Splits the file so that the data can be extracted into a ``array`` by
        ``self._get_data``

        Parameters
        ----------
        file : file
            Open file containing independent data

        Returns
        -------
        tuple
            (X, Y) where X and Y are arrays of the two independent variables
        """

        # pylint: disable=inconsistent-return-statements
        # as it breaks the function when changed to return None
        def get_n_elements(line):
            for i in line.split(" "):
                try:
                    return np.int64(i)
                except ValueError:
                    pass

        for line in file:
            if "X_SIZE" in line:
                self._X_dim = get_n_elements(line)
            elif "Y_SIZE" in line:
                self._Y_dim = get_n_elements(line)
                break

        for line in file:
            if "X_COORDINATES" in line:
                _ = next(file)
                break

        file_split = iter([word for line in file for word in line.split(" ")
                           if "Y_COORDINATES" not in line])

        X = self._get_data(file_split, self._X_dim)
        Y = self._get_data(file_split, self._Y_dim)

        return X, Y

    def parse_dep_var(self, file: IO) -> np.ndarray:
        """
        Parses the dependent variables or their errors.

        Parameters
        ----------
        file : file
            Open file containing independent data

        Returns
        -------
        numpy.ndarray
            A 2d array with dimensions of the two independent variables
        """

        file_split = iter([word for line in file for word in line.split(" ")])
        dep = self._get_data(file_split, self._Y_dim, self._X_dim)
        return dep

    def _get_data(self, str_iter: Iterable, *dimensions: float) -> np.ndarray:
        """
        Iterates over an iterator from a file and extracts the numerical values
        as data.

        Parameters
        ----------
        str_iter : iterator
            An iterator of str
        *dimensions
            A `float` specifying the size for every dimension of the data

        Returns
        -------
        numpy.ndarray
            An array of `float` with dimensions specified by ``*dimensions``
        """

        def get_row_data(dim):

            row_data = np.empty(dim)

            for j in range(dim):
                datum = None
                while datum is None:
                    datum = self._make_float(next(str_iter))
                row_data[j] = datum

            return row_data

        # ignore as this will not be `np.empty` by the end of the function
        var = np.empty(dimensions)  # type: ignore

        if len(dimensions) == 1:
            var = get_row_data(dimensions[0])
        else:
            for k in range(dimensions[0]):
                var[k] = get_row_data(dimensions[1])
        # For the 263K05Awat_LAMP data file the output is SQw structured such that:
        # np.shape(SQw) == (np.shape(Q), np.shape(E))
        # this is consistent with SQw as we currently calculate it from MD
        return var
