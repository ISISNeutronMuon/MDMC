"""A reader for netcdf SQw data"""
# disabling as there is a 'no Dataset in netCDF4' false linting warning for this file
# pylint: disable=no-name-in-module
import numpy as np
from netCDF4 import Dataset

from MDMC.common.constants import h_bar
from MDMC.readers.observables.obs_reader import SQwReader


class netCDF(SQwReader):

    """
    Currently only setup for parsing MMTK/nMOLDYN SQw netcdf files

    Attributes
    ----------
    file : file
        The netCDF input file
    """

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
