"""Module for observable reader abstract class"""

from abc import abstractmethod, ABC

import numpy as np

from MDMC.common import units
from MDMC.common.decorators import repr_decorator, unit_decorator
from MDMC.readers.reader import Reader

@repr_decorator('data')
class ObservableReader(Reader):

    """
    Abstract class that defines methods common to all readers for observables

    ObservableReaders are created using ObservableReaderFactory
    """

    def assign(self, observable: 'Observable'):
        # disable pylint warning about writing to the `Observable`
        #pylint: disable=protected-access
        """
        Abstract method to assign the parsed information into the `Observable`

        Parameters
        ----------
        observable : `Observable`
            An MDMC `Observable` that will be assigned the data parsed by the reader.
        """
        observable._independent_variables = self.independent_variables
        observable._dependent_variables = self.dependent_variables
        observable._errors = self.errors

    @property
    def data(self):
        """
        A dictionary of dictionaries containing the independent variables,
        dependent variables and the associated errors.

        Returns
        -------
        dict
            The independent variables, dependent variables and the errors on
            the dependent variables
        """

        return {"independent": self.independent_variables,
                "dependent": self.dependent_variables,
                "errors": self.errors}

    @property
    @abstractmethod
    def independent_variables(self):
        """
        The independent variables

        Returns
        -------
        dict
            A dictionary of the independent variables
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def dependent_variables(self):
        """
        The dependent variables

        Returns
        -------
        dict
            A dictionary of the dependent variables
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def errors(self):
        """
        The errors on the dependent variables

        Returns
        -------
        dict
            A dictionary of the errors on the dependent variables
        """

        raise NotImplementedError

    @staticmethod
    def _make_float(i):
        """
        Casts the input to a `float`, or passes if the input cannot be cast

        Parameters
        ----------
        i : numeric
            Input to be cast to `float`

        Returns
        -------
        float
            A non-negative `float`, if the input can be converted to a `float`.
        """

        try:
            return np.float64(i)
        except ValueError:
            return None


class SQwReader(ObservableReader, ABC):
    """Abstract base subclass that adds attributes & methods common to all SQw readers"""
    # pylint: disable=attribute-defined-outside-init
    # to avoid it flagging up on private attributes in getters

    def __init__(self, file_name):
        super().__init__(file_name)
        self.SQw = None
        self.SQw_err = None

    @property
    def independent_variables(self):
        """
        Get the independent variables, Q (in ``Ang^-1``) and E (``meV``)

        Returns
        -------
        dict
            The independent variables Q and E
        """

        return {"Q": self.Q, "E": self.E}

    @property
    def dependent_variables(self):
        """
        Get the dependent variables, SQw (in ``arb``)

        Returns
        -------
        dict
            The dependent variables, SQw (in ``arb``)
        """

        return {"SQw": [self.SQw]}

    @property
    def errors(self):
        """
        Get the errors on the dependent variables

        Returns
        -------
        dict
            The error on SQw (in ``arb``)
        """

        return {"SQw": [self.SQw_err]}

    @property
    def w(self):
        """
        Get or set the energy transfer expressed in angular frequency, w, in
        ``1 / ps``

        Returns
        -------
        array
            Energy transfer as angular frequency, w, in ``1 / ps``
        """

        return self._w

    @w.setter
    @unit_decorator(unit=units.Unit('ps') ** -1)
    def w(self, value):

        self._w = value

    @property
    def E(self):
        """
        Get or set the energy transfer, E, in ``meV``

        Returns
        -------
        array
            Energy transfer, E, in ``meV``
        """

        return self._E

    @E.setter
    @unit_decorator(unit=units.ENERGY_TRANSFER)
    def E(self, value):

        self._E = value

    @property
    def Q(self):
        """
        Get or set the momentum transfer, Q, in ``Ang^-1``

        Returns
        -------
        array
            Momentum transfer, Q, in ``Ang^-1``
        """

        return self._Q

    @Q.setter
    @unit_decorator(unit=units.LENGTH ** -1)
    def Q(self, value):

        self._Q = value


class PDFReader(ObservableReader, ABC):
    """Abstract base subclass that adds attributes & methods common to all PDF readers"""

    def __init__(self, file_name):
        super().__init__(file_name)
        self.PDF = None
        self.PDF_err = None

    @property
    def independent_variables(self):
        """
        Get the independent variable r (in ``Ang^-1``)

        Returns
        -------
        dict
            The independent variable r (in ``Ang^-1``)
        """

        return {"r": self.r}

    @property
    def dependent_variables(self):
        """
        Get the dependent variable PDF, the pair distribution function (in ``barn``)

        Returns
        -------
        dict
            The dependent variable, PDF (in ``barn``)
        """

        return {"PDF": self.PDF}

    @property
    def errors(self):
        """
        Get the errors on the dependent variable

        Returns
        -------
        dict
            The error on PDF (in ``barn``)
        """

        return {"PDF": self.PDF_err}

    @property
    def r(self):
        """
        Get or set the value of the atomic separation distance (in ``Ang``)
        """

        return self._r

    @r.setter
    @unit_decorator(unit=units.Unit('Ang'))
    def r(self, value):

        self._r = value

    @property
    def PDF(self):

        """
        Get or set the pair/radial distribution function between pairs (in ``barn``)
        Returns
        -------
        numpy.ndarray
            pair/radial distribution function (in ``barn``)
        """

        return self._PDF

    @PDF.setter
    @unit_decorator(unit=units.Unit('barn'))
    def PDF(self, value):

        self._PDF = value

    @property
    def PDF_err(self):

        """
        Get or set the error on the pair/radial distribution function between pairs (in ``barn``)
        Returns
        -------
        numpy.ndarray
            error on the pair/radial distribution function (in ``barn``)
        """

        return self._PDF_err

    @PDF_err.setter
    @unit_decorator(unit=units.Unit('barn'))
    def PDF_err(self, value):

        self._PDF_err = value
