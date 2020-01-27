"""Module for reader abstract class"""

from abc import ABC, abstractmethod

from MDMC.common.decorators import repr_decorator

@repr_decorator('file')
class Reader(ABC):

    """
    Abstract class that defines methods common to all readers
    """

    def __init__(self):

        self.file = None

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
    def parse(self, **settings):

        """
        Parses the file data so that it is in a format expected by the class
        calling the data reader

        For readers which are not specific to one data type, the calling class
        must be determined so that the file data can be parsed into
        the appropriate data type.
        """

        raise NotImplementedError
