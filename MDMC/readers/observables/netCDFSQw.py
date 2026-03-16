"""A reader for netcdf SQw data"""
# disabling as there is a 'no Dataset in netCDF4' false linting warning for this file
# pylint: disable=no-name-in-module
import logging
from typing import Any

import numpy as np
from netCDF4 import Dataset

from MDMC.common.constants import h_bar
from MDMC.readers.observables.obs_reader import SQwReader

logger = logging.getLogger(__name__)

class netCDFSQw(SQwReader):

    """
    Currently only setup for parsing MMTK/nMOLDYN SQw netcdf files

    Attributes
    ----------
    file : file
        The netCDF input file
    """

    def __enter__(self) -> None:
        """
        Opens the file for parsing
        """

        self.file = Dataset(self.file_name, 'r', encoding='UTF-8')

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """Closes the file after parsing"""

        self.file.close()

    def parse(self, **settings: Any) -> None:
        """
        Parse into SQw format

        E is the energy transfer (in ``meV``)
        Q is wavevector transfer (in ``Ang^-1``)
        """
        # Convert hbar (eV*s) to meV*s
        # Convert angular_frequency (Thz) to Hz
        # Units cancel out to meV
        self.E = ((np.array(self.file.variables['angular_frequency']) * 1e3) *
                  (1e12 * h_bar))

        Q = self.file.variables['q']
        # nMOLDYN uses nm, so we have to convert to Ang for use in MDMC
        if 'nm' in Q.units:
            Q = np.array(Q) * 0.1
        self.Q = np.array(Q)

        self.SQw = np.abs(np.array(self.file.variables['Sqw-total']))
        self.SQw_err = np.power(np.abs(self.SQw), 0.5)

        if np.any(self.SQw_err <= 0.):
            self.SQw_err[np.where(self.SQw_err <= 0.)] = float('inf')
            msg = "We have set the error bar to infinity for any zero error values, this allows\
                us to calculate chi-squared but effectively ignores these points, this may not\
                be what you want to do, consider using a FoM which doesn't need errors if\
                this is an issue"
            logger.warning(msg)
