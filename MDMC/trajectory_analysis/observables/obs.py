"""Module defining a class for storing, calculating and reading in observables
from molecular dynamics trajectories.

AUTHOR :    Thomas Farmer        START DATE :    2018-4-26 10:14:51"""

from abc import ABCMeta, abstractmethod, abstractproperty

from MDMC.readers.reader_factory import ReaderFactory

class Observable:

    """
    Abstract class that defines methods common to all observable
    data containers

    Observable data can either be from a file or calculated from
    MD and stored in the data property, along with the associated uncertainty.
    The boolean property from_MD states the source of the information.
    """

    __metaclass__ = ABCMeta

    @property
    def name(self):

        """
        The module name that was used for factory instantiation
        """

        return self._name

    @name.setter
    def name(self, name):

        self._name = name

    @property
    def origin(self):

        """
        The origin of the observable: experiment, MD, difference
        """

        return self._origin

    @origin.setter
    def origin(self, origin):

        self._origin = origin

    @abstractproperty
    def data(self):

        pass

    @abstractproperty
    def independent_variables(self):

        """
        Return:
        Dictionary of independent variables
        """

        pass

    @abstractproperty
    def dependent_variables(self):

        """
        Return:
        Dictionary of dependent variables
        """

        pass

    @abstractproperty
    def errors(self):

        """
        Return:
        Dictionary of errors on the dependent variables
        """

        pass

    # TODO: Potentially the reader can be selected based upon file name and experimental observable class type
    def read_from_file(self, reader, file_name):

        """
        Reads in experimental data from a file using a specified reader
        """

        self._origin = 'experiment'
        self.reader = ReaderFactory.create_reader(reader)
        self.reader.open(file_name)
        self.reader.parse()

    # TODO: Currently uses the generic parameter MD_input - if this is only histograms then change this
    @abstractmethod
    def calculate_from_MD(self, MD_input, **params):

        """
        Calculates the obseravable using input from an MD simulation

        Arguments:
        MD_input - some input from an MD simulation, commonly a trajectory
        Params:
        additional parameters required for calculation to be passed e.g.
        variables for RDF prefactor calculation, independent variable axis
        """

        pass
