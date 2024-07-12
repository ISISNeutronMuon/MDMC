"""
A reader for netCDF SQw data.
"""
# disabling as there is a 'no Dataset in netCDF4' false linting warning for this file
# pylint: disable=no-name-in-module
import logging

import numpy as np
from netCDF4 import Dataset

from MDMC.common.constants import h_bar
from MDMC.readers.observables.obs_reader import SQwReader

logger = logging.getLogger(__name__)


class netCDFSQw(SQwReader):
    """
    Class to handle netCDF format SQW files.

    Parameters
    ----------
    file_name : str
        File to read data from.

    Attributes
    ----------
    file : ~typing.IO
        The netCDF input file
    SQw : ~numpy.ndarray, size(Q) x size(E)
        2D array of intensity of ``S``
    SQw_err : ~numpy.ndarray, size(Q) x size(E)
        2D array of error in ``S``
    Q : ~numpy.ndarray
        1D array of wavevector transfer (in ``Ang^-1``).
    E : ~numpy.ndarray
        1D array of energy transfer (in ``meV``).

    Notes
    -----
    Currently only setup for parsing MMTK/nMOLDYN SQw netcdf files.
    """

    def __enter__(self) -> None:
        """
        Open the file for parsing.
        """
        self.file = Dataset(self.file_name, 'r', encoding='UTF-8')

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """
        Close the file after parsing.

        Parameters
        ----------
        exception_type : Type[BaseException]
            Type of exception raised.
        exception_value : BaseException
            The exception itself.
        traceback : TraceBackType
            Traceback from error.
        """
        self.file.close()

    def parse(self, **settings: dict) -> None:
        """
        Parse into SQw format.

        Parameters
        ----------
        **settings : dict
            No extra options used in this reader.
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
            logger.warning(self.SQW_ERR_WARNING)
