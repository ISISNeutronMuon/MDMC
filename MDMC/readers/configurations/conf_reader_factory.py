"""Factory class for generating readers for configurations"""

from MDMC.readers.reader_factory import ReaderFactory
from MDMC.readers.configurations.conf_reader import ConfigurationReader


class ObservableReaderFactory(ReaderFactory):

    """
    Provides a factory for creating readers.  Any module within the readers
    submodule can be created with a string of the class name, as long as it is a
    subclass of Reader.
    """

    @staticmethod
    def base_class():

        return ConfigurationReader
