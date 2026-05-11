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

"""
Module for exporter abstract class.
"""

from abc import ABC, abstractmethod
from typing import Any

from MDMC.common.decorators import repr_decorator


@repr_decorator("file")
class Exporter(ABC):
    """
    Abstract context manager class that defines methods common to all exporters.

    Parameters
    ----------
    file_name : str
        Name of file to export.
    """

    def __init__(self, file_name: str):
        self.file = None
        self.file_name = file_name

    def __enter__(self) -> None:
        """
        Provide a generic implementation of file opening using inbuilt pythond `open`.

        Should be overridden if necessary for specific file types.
        """
        # pylint: disable=consider-using-with
        # as this is an abstracted open method
        self.file = open(self.file_name, "w", encoding="UTF-8")

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """
        Close all three files after parsing.

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
    def write(self, obj, **settings: Any) -> None:
        """
        Write the file data in the correct format.

        For exporters which are not specific to one data type, the calling class
        must be determined so that the file data can be parsed into
        the appropriate data type.

        Parameters
        ----------
        obj : Any
            The object to be exported.
        **settings : dict
            Dictionary of settings for exporter.
        """

        raise NotImplementedError
