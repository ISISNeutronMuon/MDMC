"""Reader for radial distribution functions"""

import numpy as np

from MDMC.readers.observables.obs_reader import PDFReader


class LAMPPDF(PDFReader):
    """
    A class for reading files from LAMP that contain pair/radial distribution function data
    LAMP's ascii output uses a single file, with the expected file structure being:
    Row-Number  Distance  rdf1  rdf2  ...  rdfN

    Attributes
    ----------
    file_name : file
        File containing the pair/radial distribution function data
    """

    def __init__(self, file_name):
        super().__init__(file_name)
        self._numpdfs = None

    def parse(self, **settings):

        """
        Parse the file into format for pair/radial distribution functions from LAMP

        r is the radial distance (in ????nm????) CHECK WHAT CORRECT THE UNITS ARE!!!
        PDF is the pair/radial distribution function (in barn)

        """
        pdf_array = []
        for i, line in enumerate(self.file):
            if i == 3:
                #the 4th line contains information on the time-step and number of rows/distances
                columns = line.strip().split()
                r_array = np.zeros(int(columns[1]))
            elif i > 3:
                columns = line.strip().split()
                # r is the radial distance in nm so convert to Angstrom !!! CHECK !!!
                r_array[i - 4] = float(columns[1])
                # columns 3 onwards are the pair/radial distribution functions (in barn)
                pdf_array.append([float(value) for value in columns[2:]])

        self.r = r_array
        self.PDF = np.array(pdf_array)
        self.PDF_err = np.zeros(np.shape(pdf_array))
