"""
A reader for netcdf PDF data.
"""
import re

# disabling as there is a 'no Dataset in netCDF4' false linting warning for this file
# pylint: disable=no-name-in-module
import numpy as np
from netCDF4 import Dataset

from MDMC.readers.observables.obs_reader import PDFReader


class netCDFPDF(PDFReader):
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
    r : ~numpy.ndarray
        The radial distance (in ``Ang``).
    PDF : ~numpy.ndarray
        The total pair distribution function (in ``barn``).
    PDF_err : ~numpy.ndarray
        The error in the total pair distribution function (in ``barn``).
    partial_pdfs : dict[str, ~numpy.ndarray]
        The partial PDFs (in ``barn``),
        imported from the remaining columns with the atomic labels.

    Notes
    -----
    Currently only setup for parsing MMTK/nMOLDYN SQw netcdf files.
    """

    def __enter__(self) -> None:
        """
        Open the file for parsing.
        """

        self.file = Dataset(self.file_name, 'r', encoding="UTF-8")

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
        Parse into PDF format.

        Parameters
        ----------
        **settings : dict
            No extra options used in this reader.
        """
        # Scale units as nMOLDYN uses nm, rather than Ang
        self.r = np.array(self.file.variables['r'][:]) * 10.
        self.PDF = np.array(self.file.variables['pdf-total'][:])
        self.extract_partial_pdf()
        # No errors detailed in nMOLDYN netCDF PDF file - replacing with zeroes
        self.PDF_err = np.zeros(len(self.file.variables['r']))

    def extract_partial_pdf(self) -> None:
        """
        Get partial PDFs from file.

        Automatically detects the partial PDF names within the file
        and extracts them nMOLDYN saves partial pdfs in the following
        format: "pdf-[element1]-[element2]"
        """
        # Intermediate value need as partial_PDFs can only be set as a full dict value
        intermediate_dict = {}
        pattern = re.compile("pdf-.{1,2}-.{1,2}")
        for var in self.file.variables:
            if re.fullmatch(pattern, var):
                intermediate_dict[var] = np.array(self.file.variables[var][:])

        self.partial_pdfs = intermediate_dict
