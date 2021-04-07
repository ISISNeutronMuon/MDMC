"""Module defining a class for storing, calculating and reading in observables
from molecular dynamics trajectories."""

from abc import ABC, abstractmethod

from MDMC.common.decorators import repr_decorator
from MDMC.readers.observables.obs_reader_factory import ObservableReaderFactory


@repr_decorator('origin', 'data')
class Observable(ABC):

    """
    Abstract class that defines methods common to all observable
    data containers

    Observable data can either be from a file or calculated from
    MD and stored in the data property, along with the associated uncertainty.
    The `bool` property ``from_MD`` states the source of the information.

    Attributes
    ----------
    reader : ObservableReader
        The file reader for reading experimental data
    """

    @property
    def name(self):

        """
        Get or set the module name that was used for factory instantiation

        Returns
        -------
        str
            The name of the module in which the ``Observable`` is located
        """

        return self._name

    @name.setter
    def name(self, name):

        self._name = name

    @property
    def origin(self):

        """
        Get or set the origin of the observable

        Returns
        -------
        str
            The origin of the ``Observable``, either ``'experiment'`` or
            ``'MD'``
        """

        return self._origin

    @origin.setter
    def origin(self, origin):

        self._origin = origin

    @property
    @abstractmethod
    def data(self):

        """
        The independent, dependent and error data in the Observable

        Returns
        -------
        dict
            The independent, dependent and error data
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def independent_variables(self):

        """
        The independent variables

        Return
        ------
        dict
            The independent variables
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def dependent_variables(self):

        """
        The dependent variables

        Return
        ------
        dict
            The dependent variables
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def errors(self):

        """
        The errors on the dependent variables

        Return
        ------
        dict
            The errors
        """

        raise NotImplementedError

    def read_from_file(self, reader, file_name):

        """
        Reads in experimental data from a file using a specified reader

        Parameters
        ----------
        reader : str
            The name of the required file reader
        file_name : str
            The name of the file
        """

        self._origin = 'experiment'
        self.reader = ObservableReaderFactory.create_reader(reader)
        self.reader.open(file_name)
        self.reader.parse()
        self._independent_variables = self.reader.independent_variables
        self._dependent_variables = self.reader.dependent_variables
        self._errors = self.reader.errors

    @abstractmethod
    def calculate_from_MD(self, MD_input, **parameters):

        """
        Calculates the obseravable using input from an MD simulation

        Parameters
        ----------
        MD_input : Object
            Some input from an MD simulation, commonly a ``Trajectory``
        **parameters
            Additional parameters required for calculation specific
            ``Observable`` objects
        """

        raise NotImplementedError
