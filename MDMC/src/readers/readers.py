"""Module for reader abstract class

AUTHOR :    Thomas Farmer        START DATE :    2018-6-5 17:23:06"""

from abc import ABCMeta, abstractmethod, abstractproperty

class Reader:

    """
    Abstract class that defines methods common to all readers
    """

    # TODO: Consider if splitting these methods may cause issue - should I be using a with ... as statement?
    def open(self, file_name):

        """
        Provides a generic implementation of file opening using inbuilt python
        open

        Should be overriden if necessary for specific file types.
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

    @abstractproperty
    def data(self):

        raise NotImplementedError
