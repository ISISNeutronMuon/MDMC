"""
Readers for dynamic data.
"""

import logging
from typing import IO, Iterable

import numpy as np

from MDMC.readers.observables.obs_reader import SQwReader

logger = logging.getLogger(__name__)


class LAMPSQw(SQwReader):
    """
    A class for reading SQw files from LAMP.

    LAMP's ASCII output uses three files:

    - One for independent variables and parameters (``<file_name>``)
    - another for dependent variables (``<file_name>ascii``)
    - and a third for the errors in the dependent variables (``<file_name>ascii_e``)

    Parameters
    ----------
    file_name : str
        Base name to load data from.

    Attributes
    ----------
    file_indep : ~typing.IO
        File containing the independent variables.
    file_dep : ~typing.IO
        File containing the dependent variables.
    file_dep_err: ~typing.IO
        File containing the errors on the dependent variables.
    SQw : ~numpy.ndarray, size(Q) x size(E)
        2D array of intensity of ``S``
    SQw_err : ~numpy.ndarray, size(Q) x size(E)
        2D array of error in ``S``
    Q : ~numpy.ndarray
        1D array of wavevector transfer (in ``Ang^-1``).
    w : ~numpy.ndarray
        1D array of frequency (in ``ps^-1``).
    E : ~numpy.ndarray
        1D array of  energy transfer (in ``meV``).
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
        Open sources.

        Open the files for independent variables, dependent
        variables and errors on the dependent variables.
        """
        # pylint: disable=consider-using-with
        # as this is an abstracted open method

        self.file_indep = open(self.file_name, encoding='UTF-8')
        self.file_dep = open(self.file_name + 'ascii', encoding='UTF-8')
        self.file_dep_err = open(self.file_name + 'ascii_e', encoding='UTF-8')

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """
        Close all three files after parsing.

        Parameters
        ----------
        exception_type : Type[BaseException]
            Type of exception raised.
        exception_value : BaseException
            The exception itself.
        traceback : TraceBackType
            Traceback from error.
        """

        self.file_indep.close()
        self.file_dep.close()
        self.file_dep_err.close()

    def parse(self, **settings: dict) -> None:
        """
        Parse into SQw format.

        Parameters
        ----------
        **settings : dict
            Extra options.
        """

        self.E, self.Q = self.parse_indep_var(self.file_indep)
        self.SQw = self.parse_dep_var(self.file_dep)
        self.SQw_err = self.parse_dep_var(self.file_dep_err)

        # LAMP sets errors -1 if the corresponding datum is 0.  Change these to
        # inf so that error calculations can still be performed on them but
        # result in inf.
        if np.any(self.SQw_err <= 0.):
            self.SQw_err[np.where(self.SQw_err <= 0.)] = float('inf')
            logger.warning(self.SQW_ERR_WARNING)

    def parse_indep_var(self, file: IO) -> tuple[np.ndarray, np.ndarray]:
        """
        Parse the independent variables.

        Splits the file so that the data can be extracted into a ``array`` by
        ``self._get_data``.

        Parameters
        ----------
        file : IO
            Open file containing independent data.

        Returns
        -------
        tuple[~numpy.ndarray, ~numpy.ndarray]
            (X, Y) where X and Y are arrays of the two independent variables.
        """

        def get_n_elements(line: str) -> np.int64:
            """
            Get number of elements in line.

            Parameters
            ----------
            line : str
                Input line to check.

            Returns
            -------
            ~numpy.int64
                Number of elements in line.
            """
            for i in line.split(" "):
                try:
                    return np.int64(i)
                except ValueError:
                    pass

            return None

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
        Parse the dependent variables or their errors.

        Parameters
        ----------
        file : ~typing.IO
            Open file containing independent data.

        Returns
        -------
        ~numpy.ndarray
            A 2d array with dimensions of the two independent variables.
        """

        file_split = iter([word for line in file for word in line.split(" ")])
        dep = self._get_data(file_split, self._Y_dim, self._X_dim)
        return dep

    def _get_data(self, str_iter: Iterable[str], *dimensions: float) -> np.ndarray:
        """
        Iterate over a `str` iterator and extract the numerical values as data.

        Parameters
        ----------
        str_iter : Iterable[str]
            An iterator of str.
        *dimensions : float
            A `float` specifying the size for every dimension of the data.

        Returns
        -------
        numpy.ndarray
            An array of `float` with dimensions specified by ``*dimensions``.
        """

        def get_row_data(dim) -> np.ndarray:
            """
            Get first `dim` data from each row as `float` s.

            Parameters
            ----------
            dim : int
                Number of data to read.

            Returns
            -------
            ~numpy.ndarray
                Parsed data from line.
            """

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
