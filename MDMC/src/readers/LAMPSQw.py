"""Readers for dynamic data

AUTHOR :    Thomas Farmer        START DATE :    2018-6-6 14:38:55"""

import numpy as np

from MDMC.src.readers.readers import Reader
import MDMC.src.utilities.constants as const

# TODO: Determine if base class for dynamic data is required

class LAMPSQw(Reader):

    def open(self, file_name):

        """
        LAMP's ascii output uses three files: 1 for independent variables and
        parameters (..._LAMP), another for dependent variables
        (..._LAMPascii), and a third for the errors in the dependent variables
        (...LAMPascii_e)
        """

        self.file_indep = open(file_name)
        self.file_dep = open(file_name + 'ascii')
        self.file_dep_err = open(file_name + 'ascii_e')

    def parse(self):

        """
        Parse into SQW format
        """

        self.parse_indep_var(self.file_indep)
        self.parse_dep_var(self.file_dep)
        self.parse_dep_var(self.file_dep_err, error=True)

    # TODO: Consider if indep_var should be more explicit i.e. an ordered array or dictionary
    # TODO: Make data self descriptive, so that whatever is calling reader.data can interogate it
    @property
    def data(self):

        return np.array([self.Q, self.E, self.SQw, self.SQw_err])


    # TODO: Refactor and abstract out E and q
    def parse_indep_var(self, file):

        """
        Determines the number of elements of the independent variables and
        creates a numpy array of that size.

        file is an iterator

        X is energy transfer (E in meV)
        Y is wavevector transfer (q in AA^-1)
        """

        def get_n_elements(line):
            for i in line.split(" "):
                try:
                    return np.int64(i)
                except ValueError:
                    pass

        for line in file:
            if "X_SIZE" in line:
                self.E_dim = get_n_elements(line)
            elif "Y_SIZE" in line:
                self.Q_dim = get_n_elements(line)
                break

        for line in file:
            if "X_COORDINATES" in line:
                _ = next(file)
                break

        file_split = iter([str for line in file for str in line.split(" ")
            if "Y_COORDINATES" not in line])

        self.E = np.empty(self.E_dim)
        self.Q = np.empty(self.Q_dim)
        self._get_data(self.E, self.E_dim, file_split)
        self._get_data(self.Q, self.Q_dim, file_split)

    # TODO: Refactor to deal with errors better - DRY
    def parse_dep_var(self, file, error=False):

        file_split = iter([str for line in file for str in line.split(" ")])

        if error:
            self.SQw_err = np.empty([self.Q_dim, self.E_dim])
            for k in range(self.Q_dim):
                self._get_data(self.SQw_err[k], self.E_dim, file_split)
        else:
            self.SQw = np.empty([self.Q_dim, self.E_dim])
            for k in range(self.Q_dim):
                self._get_data(self.SQw[k], self.E_dim, file_split)

    def _make_float(self, i):
        try:
            return np.float64(i)
        except ValueError:
            pass

    def _get_data(self, var, dim, str_iter):
        for j in range(dim):
            datum = None
            while datum is None:
                datum = self._make_float(next(str_iter))
            var[j] = datum
