# MDMC is a package for the optimisation of classical potentials with experimental data
# Copyright (C) 2026 MDMC Developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Reader for pair distribution function data from LAMP's ascii files"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from MDMC.readers.observables.obs_reader import PDFReader

if TYPE_CHECKING:
    from MDMC.trajectory_analysis.observables.pdf import PairDistributionFunction


class LAMPPDF(PDFReader):
    """
    A class for reading files from LAMP that contain pair distribution function (PDF) data.
    LAMP's ascii output uses a single file, with the expected file structure being:
    Row-Number  Distance  pdf-total  pdf1 pdf2  ...  pdfN

    The column file format above is the default with the total PDF data contained in the 3 column.
    When initialising instances of this class it is possible to change this using the `pdf_col`
    parameter to select which data column contains the total PDF. The remaining columns (if they
    exist) are assumed to be partial PDFs.

    Parameters
    ----------
    file_name : str
        File containing the pair distribution function data
    pdf_col : int, optional >= 3
        Column that contains the data to be saved as the total PDF
        (`PairDistributionFunction.PDF`). Optional, default value is 3 as columns 1 and 2 are
        reserved for the row-counter and the distance value.
    partial_strings : list of tuples
        List of tuples to specify the labels of the partial pairs to be saved as such in
        `PairDistributionFunction.partial_pdfs`. All columns in the data file apart from the
        row-counter (column 1), distance values (column 2) and the one for the total PDF
        (`pdf_col`) are saved as `partial_pdfs`. The labels are applied in numerical order. If
        no labels are specified, the column header in the data file is used as the label.
    """

    def __init__(self, file_name: str, pdf_col: int = 3, partial_strings: list[tuple] = None):
        super().__init__(file_name)
        self.pdf_col = pdf_col
        self.partial_pdfs = {}
        self.partial_strings = partial_strings

    def assign(self, observable: PairDistributionFunction) -> None:
        # disable pylint warning about writing to the `Observable`
        # pylint: disable=protected-access
        """
        Method to assign the data parsed by the LAMPPDF reader to a PDF `Observable`.

        Parameters
        ----------
        observable : PairDistributionFunction
            The PairDistributionFunction to which the parsed information should be assiged.
        """
        observable._independent_variables = self.independent_variables
        observable._dependent_variables = self.dependent_variables
        observable._errors = self.errors
        observable.partial_pdfs = self.partial_pdfs
        observable.partial_strings = self.partial_strings

    def parse(self, **settings: Any) -> None:
        """
        Parse the file information

        `r` is the radial distance (in Angstrom), expected in column 2 of the file
        `PDF` is the total pair distribution function (in barn), by default expected in column 3
        of the file, but can be specified by `pdf_col` setting.
        `partial_pairs` are the partial PDFs (in barn), imported from the remaining columns with
        the labels of the partial pairs either specified by `partial_strings` or taken from the
        column headers.

        """
        pdf_array = []
        for i, line in enumerate(self.file):
            columns = line.strip().split()
            if i == 2:
                # extract column headers if needed
                if self.partial_strings is None:
                    self.partial_strings = columns[4:]
            elif i == 3:
                # the 4th line contains information on the time-step and number of rows/distances
                r_array = np.zeros(int(columns[1]))
            elif i > 3:
                r_array[i - 4] = float(columns[1])
                # columns 3 onwards are the pair distribution functions (in barn)
                pdf_array.append([float(value) for value in columns[2:]])
        pdf_array = np.array(pdf_array)

        self.r = r_array
        self.PDF = pdf_array[:, self.pdf_col - 3]
        self.PDF_err = np.zeros(np.shape(self.PDF))

        # select partial pair columns by deleting the total PDF column
        pp_array = np.delete(pdf_array, self.pdf_col - 3, axis=1)
        try:
            assert np.shape(pp_array)[1] == len(self.partial_strings)
        except AssertionError as error:
            msg = (
                f"The number of partial pair labels ({len(self.partial_strings)}) is not the "
                f"same as the number of data columns for the pairs ({np.shape(pp_array)[1]}). "
                f"This is either because the number of labels passed is incorrect or because "
                f"the column labels are not recognised correctly, e.g. due to an unexpected "
                f"delimiter."
            )
            raise AssertionError(msg) from error
        for i, string in enumerate(self.partial_strings):
            self.partial_pdfs[string] = pp_array[:, i]
