"""Module for reader abstract class"""

from abc import ABC, abstractmethod

class Reader(ABC):

    """
    Abstract class that defines methods common to all readers

    Readers are created using ReaderFactory
    """

    # TODO: Consider if splitting these methods may cause issue - should I be using a with ... as statement?
    def open(self, file_name):

        """
        Provides a generic implementation of file opening using inbuilt python
        open

        Should be overriden if necessary for specific file types.

        Parameters
        ----------
        file_name : str
            The name of the input file
        """

        self.file = open(file_name, 'r')

    @abstractmethod
    def parse(self):

        """
        Parses the file data so that it is in a format expected by the class
        calling the data reader

        For readers which are not specific to one data type, the calling class
        must be determined so that the file data can be parsed into
        the appropriate data type.
        """

        pass

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

        return {"independent":self.independent_variables,
                "dependent":self.dependent_variables,
                "errors":self.errors}

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

        pass

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

        pass

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

        pass
