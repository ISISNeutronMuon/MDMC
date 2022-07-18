"""Factory class for generating readers for observables"""

from MDMC.readers.reader_factory import ReaderFactory
from MDMC.readers.observables.obs_reader import ObservableReader


class ObservableReaderFactory(ReaderFactory):

    """
    Provides a factory for creating readers.  Any module within the readers
    submodule can be created with a string of the class name, as long as it is a
    subclass of ``Reader``.
    """

    @staticmethod
    def base_class() -> ObservableReader:

        return ObservableReader
