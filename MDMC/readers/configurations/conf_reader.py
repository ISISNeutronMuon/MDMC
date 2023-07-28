"""Module for observable reader abstract class"""

from abc import abstractmethod
from typing import TYPE_CHECKING, List

from MDMC.common.decorators import repr_decorator
from MDMC.readers.reader import Reader

if TYPE_CHECKING:
    from MDMC.MD.structures import Atom

@repr_decorator('file', 'extension', 'atoms')
class ConfigurationReader(Reader):

    """
    Abstract class (as it does not implement ``Reader.parse``) that defines
    properties common to all readers for configurations

    A ``ConfigurationReader`` is created using ``ConfigurationReaderFactory``
    """

    def __init__(self, file_name: str):

        super().__init__(file_name)
        self._atoms: List['Atom'] = []

    @property
    def atoms(self) -> 'list[Atom]':
        """
        The `Atom` objects parsed from the file.
        """

        return self._atoms
