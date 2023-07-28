"""Factory class for generating readers for configurations"""

from importlib import import_module
from inspect import getmembers, isabstract, isclass
from os.path import dirname
from pkgutil import iter_modules

from ase.io.formats import ioformats

import MDMC.readers.configurations
from MDMC.readers.reader_factory import ReaderFactory
from MDMC.readers.configurations.conf_reader import ConfigurationReader
from MDMC.readers.configurations.ase import ASEReader


class ConfigurationReaderFactory(ReaderFactory):

    """
    Provides a factory for creating readers.  Any module within the readers
    submodule can be created with a string of the class name, as long as it is a
    subclass of ``ConfigurationReader``.
    """

    @staticmethod
    def base_class() -> type['ConfigurationReader']:

        return ConfigurationReader

    @classmethod
    def create_reader_from_ext(cls, extension: str, file_name: str) -> ConfigurationReader:

        """
        Parameters
        ----------
        extension : str
            The file extension from which to initialize a subclass of
            ``ConfigurationReader``
        file_name : str
            The name of the file that you want to read.

        Returns
        -------
        ConfigurationReader
            An initialized configuration reader which has the extension
            specified by ``extension``

        Raises
        ------
        NotImplementedError
            If ``extension`` does not match the ``extension`` property of any
            subclass of ``ConfigurationReader``
        """

        readers = getmembers(MDMC.readers.configurations,
                             lambda m: isclass(m)
                             and not isabstract(m)
                             and issubclass(m, cls.base_class()))
        for reader in readers:
            if reader[1].extension == extension:
                return reader[1](file_name)

        # if no direct reader exists, try ASE reader
        if extension in ioformats:
            return ASEReader(file_name)

        raise NotImplementedError(
            f'No implemented reader is compatible with {extension} extension')
