"""
Module for reader abstract class.
"""
from abc import ABC, abstractmethod
from typing import IO

from MDMC.common.decorators import repr_decorator


@repr_decorator('file')
class Reader(ABC):
    """
    Abstract class that defines methods common to all readers.

    Parameters
    ----------
    file_name : str
        Name of file to read.
    """

    def __init__(self, file_name: str):

        self.file: IO = None
        self.file_name = file_name

    def __enter__(self) -> None:
        """
        Interface for opening file.

        Provide generic implementation of file opening using inbuilt python
        :any:`open`.

        Should be overridden if necessary for specific file types.
        """
        # pylint: disable=consider-using-with
        # as this is an abstracted open method

        self.file = open(self.file_name, 'r', encoding='UTF-8')

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """
        Close the open file after parsing.

        Parameters
        ----------
        exception_type : Type[BaseException]
            Type of exception raised.
        exception_value : BaseException
            The exception itself.
        traceback : TraceBackType
            Traceback from error.
        """

        self.file.close()

    @abstractmethod
    def parse(self, **settings: dict) -> None:
        """
        Parse the file data.

        For readers which are not specific to one data type, the
        calling class must be determined so that the file data can be
        parsed into the appropriate data type.

        Parameters
        ----------
        **settings : dict
            Dictionary of settings for reader.
        """

        raise NotImplementedError
