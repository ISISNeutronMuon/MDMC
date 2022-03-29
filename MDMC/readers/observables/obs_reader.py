"""Module for observable reader abstract class"""

from abc import abstractmethod, ABC

import numpy as np
from MDMC.common.unit_registry import UREG

from MDMC.common import units
from MDMC.common.decorators import repr_decorator
from MDMC.readers.reader import Reader



@repr_decorator('data')
class ObservableReader(Reader):

    """
    Abstract class that defines methods common to all readers for observables

    ObservableReaders are created using ObservableReaderFactory
    """

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
    def w(self, value):

        self._w = value * (UREG.ps ** -1)

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
    def E(self, value):

        self._E = value * UREG.meV

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
    def Q(self, value):

        self._Q = value * (UREG.angstrom ** -1)
