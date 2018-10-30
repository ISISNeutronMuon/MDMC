"""Module defining a class for storing, calculating and reading in observables
from molecular dynamics trajectories.

AUTHOR :    Thomas Farmer        START DATE :    2018-4-26 10:14:51"""

from abc import ABCMeta, abstractmethod, abstractproperty

from MDMC.src.readers.reader_factory import ReaderFactory

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

    @abstractproperty
    def origin(self):

        """
        The origin of the observable: experiment, MD, difference
        """

        pass

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

        self.reader = ReaderFactory.create_reader(reader)
        self.reader.open(file_name)
        self.reader.parse()

    # TODO: Currently uses the generic parameter MD_input - if this is only histograms then change this
    @abstractmethod
    def calculate_from_MD(self, MD_input, **params):

        """
        Calculates the obseravable using input from an MD
        simulation

        params enables any additional parameters required for calculation to be
        passed e.g. variables for RDF prefactor calculation, independent
        variable axis
        """

        pass

    # TODO: Implement
    def _check_identical_indep_var(self, observable):

        """
        Checks that self and observable have identical independent variables

        This check is required for calculating FoM
        """

        raise NotImplementedError

    def __sub__(self, observable):

        """
        Returns:
        An observable of the same type, with dependent data equal to the
        difference of the two observables, and errors calculated in quadrature.
        """

        # TODO: Use _check_identical_indep_var and also test that one observable is from data and the other from MD
        obs = self.__class__()
        obs._origin = "difference"

        obs._independent_variables = self._independent_variables

        obs._dependent_variables = {}
        for key in self.dependent_variables.keys():
            obs._dependent_variables[key] = self.dependent_variables[key] \
                - observable.dependent_variables[key]

        obs._errors = {}
        for key in self.errors.keys():
            obs._errors[key] = (self.errors[key] ** 2
                + observable.errors[key] ** 2) ** 0.5

        return obs
