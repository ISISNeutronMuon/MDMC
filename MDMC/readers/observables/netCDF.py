"""A reader for netcdf SQw data"""
# disabling as there is a 'no Dataset in netCDF4' false linting warning for this file
# pylint: disable=no-name-in-module
import numpy as np
from netCDF4 import Dataset

from MDMC.common import units
from MDMC.common.constants import h_bar
from MDMC.common.decorators import unit_decorator
from MDMC.readers.observables.obs_reader import ObservableReader


class netCDF(ObservableReader):

    """
    Currently only setup for parsing MMTK/nMOLDYN SQw netcdf files

    Attributes
    ----------
    file : file
        The netCDF input file
    """

    def __init__(self):
        super().__init__()
        self.SQw = None
        self.SQw_err = None

    def open(self, file_name):
        """
        Opens the file for parsing

        Parameters
        ----------
        file_name : str
            The name of the netCDF file
        """

        self.file = Dataset(file_name, 'r')

    def parse(self, **settings):
        """
        Parse into SQw format

        E is the energy transfer (in ``meV``)
        Q is wavevector transfer (in ``Ang^-1``)
        """

        self.E = (np.array(self.file.variables['angular_frequency']) * 1e15
                  * h_bar)
        Q = self.file.variables['q']
        if 'nm' in Q.units:
            Q = np.array(Q) * 0.1
        self.Q = np.array(Q)

        self.SQw = np.abs(np.array(self.file.variables['Sqw-total']))
        self.SQw_err = np.power(np.abs(self.SQw), 0.5)

    @property
    def independent_variables(self):
        """
        Get the independent variables, Q (in ``Ang^-1``) and E (``meV``)

        Returns
        -------
        dict
            The independent variables Q and E
        """

        return {"Q": self.Q, "E": self.E}

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
    def w(self):
        """
        Get or set the energy transfer expressed in angular frequency, w, in
        ``1 / ps``

        Returns
        -------
        array
            Energy transfer as angular frequency, w, in ``1 / ps``
        """

        return self._w

    @w.setter
    @unit_decorator(unit=units.Unit('ps') ** -1)
    def w(self, value):

        self._w = value

    @property
    def E(self):
        """
        Get or set the energy transfer, E, in ``meV``

        Returns
        -------
        array
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
        array
            Momentum transfer, Q, in ``Ang^-1``
        """

        return self._Q

    @Q.setter
    @unit_decorator(unit=units.LENGTH ** -1)
    def Q(self, value):

        self._Q = value
