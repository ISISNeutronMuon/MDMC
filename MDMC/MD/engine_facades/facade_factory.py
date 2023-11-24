"""Factory class for generating MD engine facades"""

from importlib import import_module
from inspect import getmembers, isabstract, isclass
from types import ModuleType

from MDMC.MD.engine_facades.facade import MDEngine

class MDEngineFacadeFactory:

    """
    Provides a factory for creating facades to ``MDEngine``.  Any facade within
    the ``engine_facades`` folder can be created with a `str` of the class
    ``name``, as long as it is a subclass of ``MDEngine``.
    """

    @staticmethod
    def create_facade(module_name: str) -> MDEngine:
        """
        Parameters
        ----------
        module_name : str
            A module name in ``engine_facades``. Aliases to these module names
            are also valid.

        Returns
        -------
        ``MDEngine``
            The specified ``MDEngine``, as determined by the ``module_name``
        """

        try:
            module = import_module('.' + module_name, __package__)
        except ImportError:
            module = MDEngineFacadeFactory.import_from_alias(module_name)

        classes = getmembers(module, lambda m: (isclass(m)
                                                and not isabstract(m)
                                                and issubclass(m, MDEngine)))

        return classes[0][1]()

    @staticmethod
    def import_from_alias(alias: str) -> ModuleType:
        """
        Converts an ``alias`` into a module name
        """

        alias = alias.lower()
        engines = ['lammps_engine', 'dlpoly_engine']
        if not alias.endswith('_engine'):
            alias += '_engine'
        if alias in engines:
            module_name = alias
        else:
            raise ImportError(f"The MD engine {alias} is not in the list of recognised engines, "
                              f"which currently comprises: {engines}")

        return import_module('.' + module_name, __package__)
