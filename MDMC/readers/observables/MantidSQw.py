"""
Readers for dynamic Mantid SQw data.
"""

import logging
from contextlib import suppress
from typing import IO

import numpy as np

from MDMC.readers.observables.obs_reader import SQwReader

logger = logging.getLogger(__name__)


class MantidSQw(SQwReader):
    """
    A class for reading SQw files from Mantid.

    Mantid's ASCII output uses one or two files, either:

    - A file containing the SQw data and error for the range of
      energy values measured at each Q with ``file_name`` or
    - A file containing the SQw data and error for the range of energy
      values measured at each detector (or group of detectors) ID
      with ``file_name`` and a file giving the momentum value
      associated with each detector (or group of detectors) ID, with
      the name given by ``file_name + '_detectors'``

    If a single file is supplied, then it is assumed that the Q values
    are included in the data, this is the typical output of Mantid
    reduced ISIS data.

    If there are two files then it is assumed that the second file
    links the detector ID's with the corresponding Q's.

    Parameters
    ----------
    file_name : str
        File to read data from.

    Attributes
    ----------
    ID_or_Q : ~typing.IO
        File containing the ID's of the detectors.
    file_detectors : ~typing.IO
        File containing the errors on the dependent variables.
    file_variables : ~typing.IO
        File containing the variables for each detector ID or .Q
    SQw : ~numpy.ndarray, size(Q) x size(E)
        2D array of intensity of ``S``.
    SQw_err : ~numpy.ndarray, size(Q) x size(E)
        2D array of error in ``S``.
    Q : ~numpy.ndarray
        1D array of wavevector transfer (in ``Ang^-1``).
    E : ~numpy.ndarray
        1D array of energy transfer (in ``meV``).

    Notes
    -----
    An example reduction script is included in
    doc/tutorials/data/water_reduction_IRIS.py
    """

    def __init__(self, file_name: str):
        super().__init__(file_name)
        self.detector_ID_or_Q = None
        self.file_detectors = None
        self.file_variables = None

    def __enter__(self) -> None:
        """
        Open the files for variables and detector momenta.
        """
        # pylint: disable=consider-using-with
        # as this is an abstracted open method

        self.file_variables = open(self.file_name, encoding='UTF-8')
        try:
            self.file_detectors = open(self.file_name + '_detectors', encoding='UTF-8')
        except FileNotFoundError:
            self.file_detectors = None

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """
        Close variable and detector files after parsing.

        Parameters
        ----------
        exception_type : Type[BaseException]
            Type of exception raised.
        exception_value : BaseException
            The exception itself.
        traceback : TraceBackType
            Traceback from error.
        """

        self.file_variables.close()
        with suppress(AttributeError):
            self.file_detectors.close()

    def parse(self, **settings: dict) -> None:
        """
        Parse into SQw format.

        Parameters
        ----------
        **settings : dict
            No extra options used in this reader.
        """

        self.E, self.SQw, self.SQw_err = self.parse_variables(
            self.file_variables)
        if self.file_detectors is not None:
            self.Q = self.parse_detectors(self.file_detectors)
        else:
            self.Q = self._make_float(self.detector_ID_or_Q)

        # Explicitly sort data
        E_argsort = self.E.argsort()
        Q_argsort = self.Q.argsort()
        self.E = self.E[E_argsort]
        self.Q = self.Q[Q_argsort]
        self.SQw = self.SQw[Q_argsort, :]
        self.SQw = self.SQw[:, E_argsort]
        self.SQw_err = self.SQw_err[Q_argsort, :]
        self.SQw_err = self.SQw_err[:, E_argsort]

        # Mantid sets errors to 0 if the corresponding datum is 0.  Change these to
        # inf so that error calculations can still be performed on them.
        if np.any(self.SQw_err <= 0.):
            self.SQw_err[np.where(self.SQw_err <= 0.)] = float('inf')
            logger.warning(self.SQW_ERR_WARNING)

    def parse_variables(self, file: IO) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse variables from file.

        Parses the values for energy, SQw and its error for each
        detector, or momentum value if it is defined instead of
        detector_ID

        Parameters
        ----------
        file : ~typing.IO
            Open file containing the variables.

        Returns
        -------
        tuple[~numpy.ndarray, ~numpy.ndarray, ~numpy.ndarray]
            (X, Y, E) where:

            - ``X`` is the independent variable (energy),
            - ``Y`` is the dependent variable (SQw) and
            - ``E`` is the errors of ``Y``.
        """

        self.detector_ID_or_Q = []
        data = []
        for line in file:
            line = line.strip()
            # Skip any lines which are comments or headers
            if line[0] == '#':
                continue

            strings = line.split(',')
            if len(strings) == 1:
                self.detector_ID_or_Q.append(strings[0])
                data.append({'X': [], 'Y': [], 'E': []})
            else:
                data[-1]['X'].append(self._make_float(strings[0]))
                data[-1]['Y'].append(self._make_float(strings[1]))
                data[-1]['E'].append(self._make_float(strings[2]))

        X = np.array(data[0]['X'])
        Y = np.zeros((len(self.detector_ID_or_Q), len(X)))
        E = np.zeros((len(self.detector_ID_or_Q), len(X)))
        for i, datum in enumerate(data):
            # X data should be the same for each detector
            assert np.all(np.array(datum['X']) == X)
            Y[i] = np.array(datum['Y'])
            E[i] = np.array(datum['E'])

        return X, Y, E

    def parse_detectors(self, file: IO) -> np.ndarray:
        """
        Parse the detector momenta values.

        Parameters
        ----------
        file : ~typing.IO
            Open file containing detector IDs and momenta.

        Returns
        -------
        numpy.ndarray
            A 1D array of momenta values.
        """

        Q = np.zeros(len(self.detector_ID_or_Q))
        for i, line in enumerate(file):
            if i == 0:
                headings = line.split(', ')
                try:
                    ID_header = 'Spectrum No'
                    spectrum_index = headings.index(ID_header)
                except ValueError as error:
                    raise ValueError(f'Detector file must have the heading "{ID_header}"') \
                        from error

                try:
                    Q_header = 'Q'
                    Q_index = headings.index(Q_header)
                except ValueError as error:
                    raise ValueError(f'Detector file must have the heading "{Q_header}"') \
                        from error
            else:
                values = line.split()
                spectrum_no = values[spectrum_index]
                Q_value = values[Q_index]
                # Ensure that we assign Q values in the same order as detector_IDs
                Q[self.detector_ID_or_Q.index(
                    spectrum_no)] = self._make_float(Q_value)

        return Q
