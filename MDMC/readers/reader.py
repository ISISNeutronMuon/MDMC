"""Module for reader abstract class"""

from abc import ABC, abstractmethod

from MDMC.common.decorators import repr_decorator

@repr_decorator('file')
class Reader(ABC):

    """
    Abstract class that defines methods common to all readers
    """

    def __init__(self, file_name):

        self.file = None
        self.file_name = file_name

    def __enter__(self):

        """
        Provides a generic implementation of file opening using inbuilt python
        open

        Should be overridden if necessary for specific file types.
        """

        self.file = open(self.file_name, 'r', encoding='UTF-8')

    def __exit__(self, exception_type, exception_value, traceback):
        """Closes the open file after parsing"""

        self.file.close()

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
