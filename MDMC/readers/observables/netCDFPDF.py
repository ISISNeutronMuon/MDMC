"""A reader for netcdf PDF data"""
import re
from typing import Any

# disabling as there is a 'no Dataset in netCDF4' false linting warning for this file
# pylint: disable=no-name-in-module
import numpy as np
from netCDF4 import Dataset

from MDMC.readers.observables.obs_reader import PDFReader


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

        self.file = Dataset(self.file_name, 'r', encoding="UTF-8")

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """Closes the file after parsing"""

        self.file.close()

    def parse(self, **settings: Any) -> None:
        """
        Parse into PDF format
        """
        # Scale units as nMOLDYN uses nm, rather than Ang
        self.r = np.array(self.file.variables['r'][:]) * 10.
        self.PDF = np.array(self.file.variables['pdf-total'][:])
        self.extract_partial_pdf()
        # No errors detailed in nMOLDYN netCDF PDF file - replacing with zeroes
        self.PDF_err = np.zeros(len(self.file.variables['r']))

    def extract_partial_pdf(self) -> None:
        """
        Automatically detects the partial PDF names within the file and extracts them
        nMOLDYN saves partial pdfs in the following format: "pdf-[element1]-[element2]"
        """
        # Intermediate value need as partial_PDFs can only be set as a full dict value
        intermediate_dict = {}
        pattern = re.compile("pdf-.{1,2}-.{1,2}")
        for var in self.file.variables:
            if re.fullmatch(pattern, var):
                intermediate_dict[var] = np.array(self.file.variables[var][:])

        self.partial_pdfs = intermediate_dict
