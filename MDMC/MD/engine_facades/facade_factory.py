"""Factory class for generating MD engine facades"""

from importlib import import_module
from inspect import isclass, isabstract, getmembers

from MDMC.MD.engine_facades.facade import MDEngine

class MDEngineFacadeFactory(object):

    """
    Provides a factory for creating facades to MD engines.  Any facade within
    the engine_facades folder can be created with a string of the class name, as
    long as it is a subclass of MDEngine.
    """

    @staticmethod
    def create_facade(module_name):

        """
        Arguments:
        module_name - a string specifying a module name in engine_facades.
        Aliases to these module names are also valid.

        Returns:
        an object of a MD engine
        """

        try:
            module = import_module('.' + module_name, __package__)
        except ImportError:
            module = MDEngineFacadeFactory._import_from_alias(module_name)

        classes = getmembers(module, lambda m: (isclass(m)
                                                and not isabstract(m)
                                                and issubclass(m, MDEngine)))

        return classes[0][1]()

    @staticmethod
    def _import_from_alias(alias):

        """
        Converts an alias into a module name
        """

        if alias.upper() == 'MMTK':
            module_name = 'mmtk'
        elif alias.upper() == 'LAMMPS' or alias.lower() == 'lammps_engine':
            module_name = 'lammps_engine'

        return import_module('.' + module_name, __package__)
