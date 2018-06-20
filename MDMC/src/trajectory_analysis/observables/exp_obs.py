"""Module defining a class for storing, calculating and reading in experimental
observables from molecular dynamics trajectories.

AUTHOR :    Thomas Farmer        START DATE :    2018-4-26 10:14:51"""

from abc import ABCMeta, abstractmethod, abstractproperty

from MDMC.src.readers.reader_factory import ReaderFactory

class ExperimentalObservable:

    """
    Abstract class that defines methods common to all experimental observable
    data containers

    Experimental observable data can either be from a file or calculated from
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

    @abstractproperty
    def from_MD(self):

        pass

    @abstractproperty
    def data(self):

        pass

    @abstractproperty
    def independent_variables(self):

        pass

    # TODO: Determine if this can be concrete rather than abstract
    @abstractproperty
    def dependent_variables(self):

        pass

    # TODO: Potentially the reader can be selected based upon file name and experimental observable class type
    def read_from_file(self, reader, file_name):

        """
        Reads in experimental data from a file using a specified reader
        """

        self.reader = ReaderFactory.create_reader(reader)
        self.reader.open(file_name)
        self.reader.parse()

    # TODO: Currently uses the generic parameter MD_input - if this is only histograms then change this
    @abstractmethod
    def calculate_from_MD(self, MD_input, **params):

        """
        Calculates the experimental obseravable using input from an MD
        simulation

        params enables any additional parameters required for calculation to be
        passed e.g. variables for RDF prefactor calculation, independent
        variable axis
        """

        pass
