"""Factory class for generating force fields"""

from glob import glob
from importlib import import_module
from inspect import getmembers, isabstract, isclass
from os.path import basename, dirname, isfile, join

from MDMC.MD.force_fields.ff import ForceField


class ForceFieldFactory:

    """
    Provides a factory for creating a ``ForceField``.  Any force field within
    the force fields folder can be created with a string of the class name, as
    long as it is a subclass of ``ForceField``.
    """

    @staticmethod
    def create_force_field(module_name):
        """
        Returns
        -------
        ForceField
            A ``ForceField`` specified by ``module_name``
        """

        try:
            module = import_module('.' + module_name, __package__)
        except ImportError as error:
            raise ValueError(
                f'{module_name} is not a supported force field') from error
        classes = getmembers(module, lambda m: (isclass(m)
                                                and not isabstract(m)
                                                and issubclass(m, ForceField)))
        return classes[0][1]()

    @staticmethod
    def get_force_field_names():
        """
        Get the names of available force fields

        Requires all ``ForceField`` derived classes to be in modules of the same
        name

        Returns
        -------
        list
            A `list` of `str` with the names of the available ``ForceField``
            objects
        """

        force_field_names = []
        for full_module_name in glob(join(dirname(__file__), "*.py")):
            if isfile(full_module_name) and full_module_name != __file__:
                module_name = basename(full_module_name)
                if not module_name.startswith('_') and module_name != 'ff.py':
                    force_field_names.append(module_name.replace('.py', ''))

        return force_field_names
