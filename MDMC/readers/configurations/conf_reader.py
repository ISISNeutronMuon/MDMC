"""Module for observable reader abstract class"""

from abc import abstractmethod

from MDMC.readers.reader import Reader


class ConfigurationReader(Reader):

    """
    Abstract class (as it does not implement Reader.parse) that defines
    properties common to all readers for configurations

    ConfigurationReaders are created using ConfigurationReaderFactory
    """

    @property
    @staticmethod
    @abstractmethod
    def extension():

        """
        The expected file extension for the ConfigurationReader
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def atoms(self):

        """
        All subclasses must implement atoms, which returns a list of `Atom`
        objects from the data read from the file
        """

        raise NotImplementedError
