"""Factory class for generating readers"""

from abc import ABC, abstractmethod
from importlib import import_module
from inspect import getmembers, getmodule, isabstract, isclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from MDMC.readers.reader import Reader

class ReaderFactory(ABC):

    """
    Provides a abstract base class for methods and properties common to all
    reader factories
    """

    @classmethod
    def create_reader(cls, module_name: str, file_name: str) -> 'Reader':
        """
        Creates a reader object from a module name

        The reader object must be the first class in the module. The base class
        which the reader must inherit is defined by ``cls.base_class``

        Parameters
        ----------
        module_name : str
            The name of the module where the reader is the first class
        file_name : str
            The name of the file that the Reader will read

        Returns
        -------
        Reader
            A ``Reader`` object
        """

        try:
            module = import_module('.' + module_name,
                                   getmodule(cls).__package__)
        except ImportError:
            module = import_module('.' + cls._name_from_alias(module_name),
                                   getmodule(cls).__package__)

        classes = getmembers(module, lambda m: (isclass(m)
                                                and not isabstract(m)
                                                and issubclass(m,
                                                               cls.base_class(),
                                                               )))

        return classes[0][1](file_name)

    @staticmethod
    @abstractmethod
    def base_class():
        """
        This should be implemented to return the base class of objects returned
        by the ``ReaderFactory``
        """

        raise NotImplementedError

    @staticmethod
    def _name_from_alias(alias: str) -> str:
        """
        Converts an ``alias`` into a module name
        """

        return alias.lower()
