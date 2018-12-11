"""A reader for netcdf SQw data

 AUTHOR :    Thomas Farmer        START DATE :    18/09/2018, 14:40:50"""

import numpy as np
from netCDF4 import Dataset

from MDMC.readers.readers import Reader
from MDMC.common.constants import h_bar

class netCDF(Reader):

    """
    Currently only setup for parsing MMTK/nMOLDYN SQw netcdf files
    """

    def open(self, file_name):

        """
        Opens the file for parsing
        """

        self.file = Dataset(file_name, 'r')

    def parse(self):

        """
        Parse into SQw format

        E is the energy transfer (in meV)
        Q is wavevector transfer (in AA^-1)
        """

        self.E = np.array(self.file.variables['angular_frequency']) * 1e15 * h_bar
        Q = self.file.variables['q']
        if 'nm' in Q.units:
            Q =  np.array(Q) * 0.1
        self.Q = np.array(Q)

        self.SQw = np.abs(np.array(self.file.variables['Sqw-total']))
        self.SQw_err = np.power(np.abs(self.SQw), 0.5)

    @property
    def independent_variables(self):

        """
        A dictionary containing Q (in AA^-1) and E (meV)
        """

        return {"Q":self.Q, "E":self.E}

    @property
    def dependent_variables(self):

        """
        A dictionary containing SQw (in arb)
        """

        return {"SQw":self.SQw}

    @property
    def errors(self):

        """
        A dictionary containing the error associated with SQw (in arb)
        """

        return {"SQw":self.SQw_err}
