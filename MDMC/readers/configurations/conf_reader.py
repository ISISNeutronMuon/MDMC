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
    @abstractmethod
    def configuration(self):

        """
        All subclasses must implement configuration, which returns a
        Configuration object from the data read from the file
        """

        raise NotImplementedError
