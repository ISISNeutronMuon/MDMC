"""Readers for dynamic data

AUTHOR :    Thomas Farmer        START DATE :    2018-6-6 14:38:55"""

import numpy as np

from MDMC.src.readers.readers import Reader

# TODO: Determine if base class for dynamic data is required

class LAMPSQW(Reader):

    def open(self, file_name):

        """
        LAMP's ascii output uses three files: 1 for independent variables and
        parameters (..._LAMP), another for dependent variables
        (..._LAMPascii), and a third for the errors in the dependent variables
        (...LAMPascii_e)
        """

        self.file_indep = open(file_name)
        self.file_dep = open(file_name + 'ascii')
        self.file_err = open(file_name + 'ascii_e')

    def parse(self):

        """
        Parse into SQW format
        """

        self.parse_indep_var(self.file_indep)
        self.parse_dep_var(self.file_dep)

    # TODO: Refactor and abstract out E and q
    def parse_indep_var(self, file):

        """
        Determines the number of elements of the independent variables and
        creates a numpy array of that size.

        file is an iterator

        X is Energy transfer (E)
        Y is Wavevector transfer (q)
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
                self.q_dim = get_n_elements(line)
                break

        for line in file:
            if "X_COORDINATES" in line:
                _ = next(file)
                break

        file_split = iter([str for line in file for str in line.split(" ")
            if "Y_COORDINATES" not in line])

        self.E = np.empty(self.E_dim)
        self.q = np.empty(self.q_dim)
        self._get_data(self.E, self.E_dim, file_split)
        self._get_data(self.q, self.q_dim, file_split)

    def parse_dep_var(self, file):

        file_split = iter([str for line in file for str in line.split(" ")])

        self.SQW = np.empty([self.q_dim, self.E_dim])
        for k in range(self.q_dim):
            self._get_data(self.SQW[k], self.E_dim, file_split)

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
