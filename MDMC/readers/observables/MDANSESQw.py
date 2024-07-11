"""
Readers for MDANSE SQw data.
"""

import logging

import numpy as np

from MDMC.common.units import SYSTEM, Unit
from MDMC.readers.observables.obs_reader import SQwReader

logger = logging.getLogger(__name__)

#: Conversion ratio between `eV` and `J`.
eV_in_Joules = 1.602176634 * 10**(-19)
#: Avogadro's number.
mole = 6.02214076 * 10**23

#: Unit conversions for units used in MDANSE.
conversion_to_meV = {
    'J' : 6.2415091e+21,
    'kJ' : 6.2415091e+24,
    'kcal' : 2.6114474e+25,
    'cal' : 2.6114474e+22,
    'kJ/mol' : 6.2415091e+24 / mole,
    'kcal/mol' :  2.6114474e+25 / mole,
    'J/mol' : 6.2415091e+21 / mole,
    'cal/mol' :  2.6114474e+22 / mole,
    'rad/ps' : 0.6582231395941951,
    'rad/fs' : 0.6582231395941951 *1e3,
    'rad/ns' : 0.6582231395941951 *1e-3,
    '1/ps' : 0.6582231395941951 * 2*np.pi,
    '1/fs' : 0.6582231395941951 *1e3 * 2*np.pi,
    '1/ns' : 0.6582231395941951 *1e-3 * 2*np.pi,
    'meV' : 1.0,
    'eV' : 1e3,
    'keV' : 1e6,
    'ueV' : 1e-3,
}


class MDANSESQw(SQwReader):
    """
    Class for reading SQw files from MDANSE's trajectory analysis.

    The output from MDANSE analysis of trajectories is a .csv file with some lines
    of comments describing the dataset and columns/rows, followed by an array
    of numbers.

    The first row and column of the array define the axes of the data,
    where the role and physical unit of each axis is described in the comment lines
    preceding the data.

    The [0,0] element of the array is always 0.0 and is not used,
    while all the remaining points are the S(Q,w) at each
    corresponding Q and w.

    Parameters
    ----------
    file_name : str
        File to read data from.

    Attributes
    ----------
    SQw : ~numpy.ndarray, size(Q) x size(E)
        2D array of intensity of ``S``
    SQw_err : ~numpy.ndarray, size(Q) x size(E)
        2D array of error in ``S``
    Q : ~numpy.ndarray
        1D array of wavevector transfer (in ``Ang^-1``).
    E : ~numpy.ndarray
        1D array of energy transfer (in ``meV``).
    file_variables : ~numpy.ndarray
        numpy array containing all the data.
    first_row : {'Q', 'E'}
        Whether first row is Q or E.
    first_column : {'Q', 'E'}
        Whether first column is Q or E.
    q_unit : Unit or str
        Units of Q in file.
    e_unit : Unit or str
        Units of E in file.
    transpose_data : bool
        Whether read data must be transposed to E, Q.
    """

    def __init__(self, file_name: str):
        super().__init__(file_name)
        self.file_variables = None
        self.first_row = 'Q'
        self.first_column = 'E'
        self.q_unit = None
        self.e_unit = None
        self.transpose_data = True

    def __enter__(self) -> None:
        """
        Open the files for variables and detector momenta.
        """
        self.file_variables = np.loadtxt(self.file_name)

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """
        Do nothing since numpy closes the file after reading anyway.

        Parameters
        ----------
        exception_type : Type[BaseException]
            Type of exception raised.
        exception_value : BaseException
            The exception itself.
        traceback : TraceBackType
            Traceback from error.
        """

    def parse_header(self):
        """
        Read the header to get the data axis labels.

        Warns
        -----
        Unrecognised unit read from file.
        """
        header = []
        value = None
        # This loop will only read the header of the file,
        # and stop as soon as it reaches the data
        with open(self.file_name, 'r', encoding='utf-8') as source:
            for line in source:
                tokens = line.split()
                if len(tokens) == 0:
                    continue
                if '#' in tokens[0]:
                    header.append(line)
                else:
                    break

        # This part will find the relevant part of the header
        # and extract the information about the axes.
        for line in header:
            if '1st' in line or 'First' in line:
                axis_signature = line.split(':')[-1]
                variable = axis_signature.split()[0]
                unit = axis_signature.split()[1].strip("()")
                unit.replace('ang', 'Ang')  # we need this since Unit cannot handle 'ang'
                if variable == 'q':
                    value = 'Q'
                    q_unit = Unit(unit)
                    try:
                        _ = q_unit.conversion_factor
                    except KeyError:
                        logger.warning('Unit %s not recognised, replaced with 1/Ang', str(unit))
                        q_unit = Unit('1')/SYSTEM["LENGTH"]
                    self.q_unit = q_unit
                elif variable == 'omega':
                    value = 'E'
                    try:
                        conversion_to_meV[unit]
                    except KeyError:
                        self.e_unit = 'arb. u.'
                    else:
                        self.e_unit = unit
                else:
                    raise ValueError(f"Unknown variable ({variable}).")
            if "row:" in line:
                self.first_row = value
            elif "column:" in line:
                self.first_column = value
        if self.first_row != 'Q':
            self.transpose_data = False

    def parse(self, **settings: dict) -> None:
        """
        Parse into SQw format.

        Create an error on SQw 1% of the value of SQw, since MDANSE
        does not yet output an error.

        .. note::

           This should be changed once it is possible to read an error.

        Parameters
        ----------
        **settings : dict
            No extra options used in this reader.
        """
        self.parse_header()

        if self.first_row == 'Q' and self.first_column == 'E':
            self.Q = self.file_variables[0, 1:]  # Entry at [0,0] is always zero
            self.E = self.file_variables[1:, 0]
        elif self.first_row == 'E' and self.first_column == 'Q':
            self.E = self.file_variables[0, 1:]
            self.Q = self.file_variables[1:, 0]

        self.Q *= self.q_unit.conversion_factor
        self.E *= conversion_to_meV[self.e_unit]

        if self.transpose_data:
            self.SQw = self.file_variables[1:, 1:].T
        else:
            self.SQw = self.file_variables[1:, 1:]

        self.SQw_err = self.SQw*0.01  # TODO: When MDANSE outputs an error, read it in

        if np.any(self.SQw <= 0.):
            self.SQw[np.where(self.SQw <= 0.)] = 0.0

        # Change and zero errors into inf so that error calculations can still be performed on them.
        if np.any(self.SQw_err <= 0.):
            self.SQw_err[np.where(self.SQw_err <= 0.)] = float('inf')
            logger.error(self.SQW_ERR_WARNING)
