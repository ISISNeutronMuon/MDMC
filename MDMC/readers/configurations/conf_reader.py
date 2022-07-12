"""Module for observable reader abstract class"""

from abc import abstractmethod
from typing import TYPE_CHECKING

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

    @property
    @staticmethod
    @abstractmethod
    def extension() -> str:
        """
        The expected file extension for the ``ConfigurationReader``
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def atoms(self) -> 'list[Atom]':
        """
        All subclasses must implement atoms, which returns a list of ``Atom``
        objects from the data read from the file
        """

        raise NotImplementedError
