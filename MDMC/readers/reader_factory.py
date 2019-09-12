"""Factory class for generating readers"""

from importlib import import_module
from inspect import isclass, isabstract, getmembers

from MDMC.readers.readers import Reader

class ReaderFactory:

    """
    Provides a factory for creating readers.  Any module within the readers
    submodule can be created with a string of the class name, as long as it is a
    subclass of Reader.
    """

    @staticmethod
    def create_reader(module_name):

        """
        Creates a reader object from a module name

        The reader object must be the first class in the module

        Parameters
        ----------
        module_name : str
            The name of the module where the reader is the first class

        Returns
        -------
        Reader
            A Reader object
        """

        module = import_module('.' + module_name, __package__)

        classes = getmembers(module, lambda m: (isclass(m)
                                                and not isabstract(m)
                                                and issubclass(m, Reader)))

        return classes[0][1]()
