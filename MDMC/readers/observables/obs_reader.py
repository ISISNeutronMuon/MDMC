"""Module for observable reader abstract class"""

from abc import abstractmethod

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
