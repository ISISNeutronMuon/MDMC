"""Module for exporter abstract class"""
import os.path
from abc import ABC, abstractmethod

from MDMC.common.decorators import repr_decorator


@repr_decorator('file')
class Exporter(ABC):

    """
    Abstract class that defines methods common to all exporters

    Parameters
    ----------
    file_name: str
        name of file to export
    """

    def __init__(self, file_name: str):

        self.file = None
        self.file_name = file_name

    def __enter__(self) -> None:
        """
        Provides a generic implementation of file opening using inbuilt python
        open

        Should be overridden if necessary for specific file types.
        """
        # pylint: disable=consider-using-with
        # as this is an abstracted open method
        if os.path.exists(self.file_name):
            self.file = open(self.file_name, 'w', encoding='UTF-8')
        else:
            self.file = open(self.file_name, 'x', encoding='UTF-8')
            self.file = open(self.file_name, 'w', encoding='UTF-8')
    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """Closes the open file after parsing"""

        self.file.close()

    @abstractmethod
    def write(self, **settings: dict) -> None:
        """
        Writes the file data into the file so that it is in a format expected by the file format

        For exporters which are not specific to one data type, the calling class
        must be determined so that the file data can be parsed into
        the appropriate data type.

        Parameters
        ----------
        **settings: dict
            dictionary of settings for exporter
        """

        raise NotImplementedError