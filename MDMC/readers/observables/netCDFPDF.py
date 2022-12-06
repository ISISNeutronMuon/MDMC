"""A reader for netcdf PDF data"""
# disabling as there is a 'no Dataset in netCDF4' false linting warning for this file
# pylint: disable=no-name-in-module
import numpy as np
from netCDF4 import Dataset

from MDMC.readers.observables.obs_reader import PDFReader
from tests.test_data.data import OBS_DATA


class netCDFPDF(PDFReader):

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

        self.file = Dataset(self.file_name, 'r')

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """Closes the file after parsing"""

        self.file.close()

    def parse(self, **settings: dict) -> None:
        """
        Parse into PDF format
        """
        # Scale units as nMOLDYN uses nm, rather than Ang
        self.r = np.array(self.file.variables['r'][:]) * 10.
