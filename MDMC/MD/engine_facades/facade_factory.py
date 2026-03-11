"""Factory class for generating MD engine facades"""

from importlib import import_module
from inspect import getmembers, isabstract, isclass
from types import ModuleType

from MDMC.MD.engine_facades.facade import MDEngine

ENGINES = {'lammps_engine': 'LAMMPSEngine',
           'dlpoly_engine': 'DLPOLYEngine',
           'null_engine': 'NullEngine',
           'openmm_engine': 'OpenMMEngine'}

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
        module_name = MDEngineFacadeFactory.standardise_alias(module_name)
        module = import_module('.' + module_name, __package__)

        classes = getmembers(module, lambda m: (isclass(m) and
                                                not isabstract(m) and
                                                issubclass(m, MDEngine) and
                                                ENGINES[module_name] in m.__name__))
        return classes[0][1]()

    @staticmethod
    def standardise_alias(alias: str) -> ModuleType:
        """
        Converts an ``alias`` into a module name
        """

        alias = alias.lower()
        if not alias.endswith('_engine'):
            alias += '_engine'

        if alias not in ENGINES:
            raise ImportError(f"The MD engine {alias} is not in the list of recognised engines, "
                              f"which currently comprises: {tuple(ENGINES.keys())}")

        return alias
