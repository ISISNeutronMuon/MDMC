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

        self.file = Dataset(file_name, 'r')

    def parse(self):

        self.E = np.array(self.file.variables['angular_frequency']) * 1e15 * h_bar
        Q = self.file.variables['q']
        if 'nm' in Q.units:
            Q =  np.array(Q) * 0.1
        self.Q = np.array(Q)

        self.SQw = np.abs(np.array(self.file.variables['Sqw-total']))
        self.SQw_err = np.power(np.abs(self.SQw), 0.5)

    @property
    def independent_variables(self):

        return {"Q":self.Q, "E":self.E}

    @property
    def dependent_variables(self):

        return {"SQw":self.SQw}

    @property
    def errors(self):

        return {"SQw":self.SQw_err}
