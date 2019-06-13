"""Module defining a class for storing, calculating and reading in observables
from molecular dynamics trajectories."""

from abc import ABCMeta, abstractmethod, abstractproperty

from MDMC.readers.reader_factory import ReaderFactory

class Observable:

    """
    Abstract class that defines methods common to all observable
    data containers

    Observable data can either be from a file or calculated from
    MD and stored in the data property, along with the associated uncertainty.
    The boolean property from_MD states the source of the information.

    Attributes
    ----------
    reader : Reader
        The file reader for reading experimental data
    """

    __metaclass__ = ABCMeta

    @property
    def name(self):

        """
        Get or set the module name that was used for factory instantiation

        Returns
        -------
        str
            The name of the module in which the Observable is located
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
            The origin of the Observable, either experiment or MD
        """

        return self._origin

    @origin.setter
    def origin(self, origin):

        self._origin = origin

    @abstractproperty
    def data(self):

        """
        The independent, dependent and error data in the Observable

        Returns
        -------
        dict
            The independent, dependent and error data
        """

        pass

    @abstractproperty
    def independent_variables(self):

        """
        The independent variables

        Return
        ------
        dict
            The independent variables
        """

        pass

    @abstractproperty
    def dependent_variables(self):

        """
        The dependent variables

        Return
        ------
        dict
            The dependent variables
        """

        pass

    @abstractproperty
    def errors(self):

        """
        The errors on the dependent variables

        Return
        ------
        dict
            The errors
        """

        pass

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
        self.reader = ReaderFactory.create_reader(reader)
        self.reader.open(file_name)
        self.reader.parse()

    @abstractmethod
    def calculate_from_MD(self, MD_input, **params):

        """
        Calculates the obseravable using input from an MD simulation

        Parameters
        ----------
        MD_input : Object
            Some input from an MD simulation, commonly a trajectory
        **params
            additional parameters required for calculation specific Observables
        """

        pass
