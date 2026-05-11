# MDMC is a package for the optimisation of classical potentials with experimental data
# Copyright (C) 2026 MDMC Developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Module for reader abstract class"""
from abc import ABC, abstractmethod
from typing import IO, Any

from MDMC.common.decorators import repr_decorator


@repr_decorator('file')
class Reader(ABC):

    """
    Abstract class that defines methods common to all readers

    Parameters
    ----------
    file_name: str
        name of file to read
    """

    def __init__(self, file_name: str):

        self.file: IO = None
        self.file_name = file_name

    def __enter__(self) -> None:
        """
        Provides a generic implementation of file opening using inbuilt python
        open

        Should be overridden if necessary for specific file types.
        """
        # pylint: disable=consider-using-with
        # as this is an abstracted open method

        self.file = open(self.file_name, 'r', encoding='UTF-8')

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """Closes the open file after parsing"""

        self.file.close()

    @abstractmethod
    def parse(self, **settings: Any) -> None:
        """
        Parses the file data so that it is in a format expected by the class
        calling the data reader

        For readers which are not specific to one data type, the calling class
        must be determined so that the file data can be parsed into
        the appropriate data type.

        Parameters
        ----------
        **settings: Any
            dictionary of settings for reader
        """

        raise NotImplementedError
