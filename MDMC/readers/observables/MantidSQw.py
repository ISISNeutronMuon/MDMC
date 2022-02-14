"""Readers for dynamic data"""

import numpy as np

from MDMC.common import units
from MDMC.common.decorators import unit_decorator
from MDMC.readers.observables.obs_reader import ObservableReader


class MantidSQw(ObservableReader):

    """
    A class for reading SQw files from Mantid

    Mantid's ascii output uses two files:
      - A file containing the SQw data and error for the range of energy values measured at each
        detector (or group of detectors) ID with ``file_name``
      - A file giving the momentum value associated with each detector (or group of detectors) ID,
        with the name given by ``file_name + '_detectors'``

    Attributes
    ----------
    file_variables : file
        File containing the variables for each detector (group) ID
    file_detectors : file
        File containing the errors on the dependent variables
    """

    def open(self, file_name):

        """
        Open the files for variables and detector momenta

        Parameters
        ----------
        file_name : str
            The variables file name, which contains the SQw, error, and energy values for each
            detector ID
        """

        self.file_variables = open(file_name)
        self.file_detectors = open(file_name + '_detectors')

    def parse(self, **settings):

        """
        Parse into SQw format

        E is the energy transfer (in meV)
        Q is wavevector transfer (in Ang^-1)
        """

        self.E, self.SQw, self.SQw_err = self.parse_variables(self.file_variables)
        self.Q = self.parse_detectors(self.file_detectors)

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
        self.SQw_err[np.where(self.SQw_err <= 0.)] = float('inf')

    @property
    def independent_variables(self):

        """
        Get the independent variables, Q (in ``Ang^-1``) and E (``meV``)

        Returns
        -------
        dict
            The independent variables Q and E
        """

        return {"Q":self.Q, "E":self.E}

    @property
    def dependent_variables(self):

        """
        Get the dependent variables, SQw (in ``arb``)

        Returns
        -------
        dict
            The dependent variables, SQw (in ``arb``)
        """

        return {"SQw": [self.SQw]}

    @property
    def errors(self):

        """
        Get the errors on the dependent variables

        Returns
        -------
        dict
            The error on SQw (in ``arb``)
        """

        return {"SQw": [self.SQw_err]}

    @property
    def E(self):

        """
        Get or set the energy transfer, E, in meV

        Returns
        -------
        numpy.ndarray
            Energy transfer, E, in ``meV``
        """

        return self._E

    @E.setter
    @unit_decorator(unit=units.ENERGY_TRANSFER)
    def E(self, value):

        self._E = value

    @property
    def Q(self):

        """
        Get or set the momentum transfer, Q, in ``Ang^-1``

        Returns
        -------
        numpy.ndarray
            Momentum transfer, Q, in ``Ang^-1``
        """

        return self._Q

    @Q.setter
    @unit_decorator(unit=units.LENGTH ** -1)
    def Q(self, value):

        self._Q = value

    def parse_variables(self, file):

        """
        Parses the values for energy, SQw and its error for each detector, but not the momentum of
        that detector.

        Parameters
        ----------
        file : file
            Open file containing the variables

        Returns
        -------
        tuple
            (X, Y, E) where X is the independent variable (energy), Y is the dependent variable
            (SQw) and E is the errors of Y
        """

        self.detector_IDs = []
        data = []
        for line in file:
            line = line.strip()
            # Skip any lines which are comments or headers
            if line[0] == '#':
                continue

            strings = line.split(',')
            if len(strings) == 1:
                self.detector_IDs.append(strings[0])
                data.append({'X':[], 'Y':[], 'E':[]})
            else:
                data[-1]['X'].append(self._make_float(strings[0]))
                data[-1]['Y'].append(self._make_float(strings[1]))
                data[-1]['E'].append(self._make_float(strings[2]))

        X = np.array(data[0]['X'])
        Y = np.zeros((len(self.detector_IDs), len(X)))
        E = np.zeros((len(self.detector_IDs), len(X)))
        for i, datum in enumerate(data):
            # X data should be the same for each detector
            assert np.all(np.array(datum['X']) == X)
            Y[i] = np.array(datum['Y'])
            E[i] = np.array(datum['E'])

        return X, Y, E

    def parse_detectors(self, file):

        """
        Parses the detector momenta values.

        Parameters
        ----------
        file : file
            Open file containing detector IDs and momenta

        Returns
        -------
        numpy.ndarray
            A 1D array of momenta values
        """

        Q = np.zeros(len(self.detector_IDs))
        data = {}
        for i, line in enumerate(file):
            if i == 0:
                headings = line.split(', ')
                try:
                    ID_header = 'Spectrum No'
                    spectrum_index = headings.index(ID_header)
                except ValueError as error:
                    raise ValueError(f'Detector file must have the heading "{ID_header}"') from error

                try:
                    Q_header = 'Q'
                    Q_index = headings.index(Q_header)
                except ValueError as error:
                    raise ValueError(f'Detector file must have the heading "{Q_header}"') from error
            else:
                values = line.split()
                spectrum_no = values[spectrum_index]
                Q_value = values[Q_index]
                # Ensure that we assign Q values in the same order as detector_IDs
                Q[self.detector_IDs.index(spectrum_no)] = self._make_float(Q_value)

        return Q

    def _make_float(self, i):

        """
        Casts the input to a `float`, or passes if the input cannot be cast

        Parameters
        ----------
        i : numeric
            Input to be cast to `float`

        Returns
        -------
        float
            A non-negative `float`, if the input can be converted to a `float`.
        """

        try:
            return np.float64(i)
        except ValueError:
            pass
