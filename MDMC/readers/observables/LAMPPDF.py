"""Reader for radial distribution functions"""

from typing import List, Tuple
import numpy as np

from MDMC.readers.observables.obs_reader import PDFReader


class LAMPPDF(PDFReader):
    """
    A class for reading files from LAMP that contain pair/radial distribution function data
    LAMP's ascii output uses a single file, with the expected file structure being:
    Row-Number  Distance  rdf1  rdf2  ...  rdfN

    Because of the ability to have multiple columns with dependent data, the parsed self.PDF
    property will be a 2D array with the second dimension being of length N (the number of
    columns containing radial/pair distribution functions).

    Parameters
    ----------
    file_name : file
        File containing the pair/radial distribution function data
    pdf_col : int>=3
        Column that contains the data to be saved as the total PDF
        (`PairDistributionFunction.PDF`). Optional, default value is 3 as columns 1 and 2 are
        normally reserved for the row-counter and the distance value.
    partial_strings : list of tuples
        List of tuples to specify the labels of the partial pairs to be saved as such in
        `PairDistributionFunction.partial_pdfs`. All columns in the data file apart from the
        row-counter (column 1), distance values (column 2) and the one for the total PDF
        (`pdf_col`) are saved as `partial_pdfs`. The labels are applied in numerical order. If
        no labels are specified, the column header in the data file is used as the label.
    """

    def __init__(self, file_name, pdf_col: int = 3, partial_strings: List[Tuple] = None):
        super().__init__(file_name)
        self.pdf_col = pdf_col
        self.partial_pdfs = {}
        self.partial_strings = partial_strings

    def assign(self, observable: 'PairDistributionFunction'):
        # disable pylint warning about writing to the `Observable`
        #pylint: disable=protected-access
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

    def parse(self, **settings):

        """
        Parse the file information

        r is the radial distance (in Angstrom)
        PDF is the pair/radial distribution function (in barn)

        """
        pdf_array = []
        for i, line in enumerate(self.file):
            columns = line.strip().split()
            if i == 2:
                #extract column headers if needed
                if self.partial_strings is None:
                    self.partial_strings = columns[4:]
            if i == 3:
                #the 4th line contains information on the time-step and number of rows/distances
                r_array = np.zeros(int(columns[1]))
            elif i > 3:
                r_array[i - 4] = float(columns[1])
                # columns 3 onwards are the pair/radial distribution functions (in barn)
                pdf_array.append([float(value) for value in columns[2:]])
        pdf_array = np.array(pdf_array)

        self.r = r_array
        self.PDF = pdf_array[:, self.pdf_col-3]
        self.PDF_err = np.zeros(np.shape(self.PDF))

        # select partial pair columns by deleting the total PDF column
        pp_array = np.delete(pdf_array, self.pdf_col-3, axis=1)
        try:
            assert np.shape(pp_array)[1] == len(self.partial_strings)
        except AssertionError as error:
            msg = (f'The number of partial pair labels ({len(self.partial_strings)}) is not the '
                   f'same as the number of data columns for the pairs ({np.shape(pp_array)[1]}). '
                   f'This is either because the number of labels passed is incorrect or because '
                   f'the column labels are not recognised correctly, e.g. due to an unexpected '
                   f'delimiter.')
            raise AssertionError(msg) from error
        for i, string in enumerate(self.partial_strings):
            self.partial_pdfs[string] = pp_array[:, i]
