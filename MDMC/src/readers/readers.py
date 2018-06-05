"""Module for reader abstract class

AUTHOR :    Thomas Farmer        START DATE :    2018-6-5 17:23:06"""

from abc import ABCMeta, abstractmethod

class Reader:

    """
    Abstract class that defines methods common to all readers
    """

    def open(self, file_name):

        """
        Provides a generic implementation of file opening using inbuilt python
        open

        Should be overriden if necessary for specific file types.
        """

        self.file = open(file_name)


    @abstractmethod
    def parse(self):

        pass
