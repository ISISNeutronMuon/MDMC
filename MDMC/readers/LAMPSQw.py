"""Readers for dynamic data

AUTHOR :    Thomas Farmer        START DATE :    2018-6-6 14:38:55"""

import numpy as np

from MDMC.common import units
from MDMC.common.decorators import unit_decorator
from MDMC.readers.readers import Reader

class LAMPSQw(Reader):

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
    file_dep_err
        File containing the errors on the dependent variables
    """

    def open(self, file_name):

        """
        Open the files for independent variables, dependent variables and errors
        on the dependent variables

        Parameters
        ----------
        file_name : str
            The independent file name, which is the base file name for the three
            files.
        """

        self.file_indep = open(file_name)
        self.file_dep = open(file_name + 'ascii')
        self.file_dep_err = open(file_name + 'ascii_e')

    def parse(self):

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
        self.SQw_err[np.where(self.SQw_err < 0.)] = np.float('inf')

    @property
    def independent_variables(self):

        """
        Get the independent variables, Q (in Ang^-1) and E (meV)

        Returns
        -------
        dict
            The independent variables Q and E
        """

        return {"Q":self.Q, "E":self.E}

    @property
    def dependent_variables(self):

        """
        Get the dependent variables, SQw (in arb)

        Returns
        -------
        dict
            The dependent variables, SQw (in arb)
        """

        return {"SQw":self.SQw}

    @property
    def errors(self):

        """
        Get the errors on the dependent variables

        Returns
        -------
        dict
            The error on SQw (in arb)
        """

        return {"SQw":self.SQw_err}

    @property
    def E(self):

        """
        Get or set the energy transfer, E, in meV

        Returns
        -------
        array
            Energy transfer, E, in meV
        """

        return self._E

    @E.setter
    @unit_decorator(unit=units.ENERGY_TRANSFER)
    def E(self, value):

        self._E = value

    @property
    def Q(self):

        """
        Get or set the momentum transfer, Q, in Ang^-1

        Returns
        -------
        array
            Momentum transfer, Q, in Ang^-1
        """

        return self._Q

    @Q.setter
    @unit_decorator(unit=units.LENGTH ** -1)
    def Q(self, value):

        self._Q = value

    def parse_indep_var(self, file):

        """
        Parses the independent variables

        Splits the file so that the data can be extracted into a numpy array by
        self._get_data

        Parameters
        ----------
        file : file
            Open file containing independent data

        Returns
        -------
        tuple
            (X, Y) where X and Y are arrays of the two independent variables
        """

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

        file_split = iter([str for line in file for str in line.split(" ")
            if "Y_COORDINATES" not in line])

        X = self._get_data(file_split, self._X_dim)
        Y = self._get_data(file_split, self._Y_dim)

        return X, Y

    def parse_dep_var(self, file):

        """
        Parses the dependent variables or their errors.

        Parameters
        ----------
        file : file
            Open file containing independent data

        Returns
        -------
        array
            A 2d array with dimensions of the two independent variables
        """

        file_split = iter([str for line in file for str in line.split(" ")])
        dep = self._get_data(file_split, self._Y_dim, self._X_dim)
        return dep

    def _make_float(self, i):

        """
        Casts the input to a float, or passes if the input cannot be cast

        Parameters
        ----------
        i : numeric
            Input to be cast to float

        Returns
        -------
        float
            A non-negative float, if the input can be converted to a float.
        """

        try:
            return np.float64(i)
        except ValueError:
            pass

    def _get_data(self, str_iter, *dims):

        """
        Iterates over an iterator from a file and extracts the numerical values
        as data.

        Parameters
        ----------
        str_iter : iterator
            An iterator of str
        *dims
            A float specifying the size for every dimension of the data

        Returns
        -------
        array
            An array of floats with dimensions specified by *dims
        """

        def get_row_data(dim):

            row_data = np.empty(dim)

            for j in range(dim):
                datum = None
                while datum is None:
                    datum = self._make_float(next(str_iter))
                row_data[j] = datum

            return row_data

        var = np.empty(dims)

        if len(dims) == 1:
            var = get_row_data(dims[0])
        else:
            for k in range(dims[0]):
                var[k] = get_row_data(dims[1])

        return var
