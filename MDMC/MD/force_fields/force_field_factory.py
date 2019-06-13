"""Factory class for generating force fields"""

from importlib import import_module
from inspect import isclass, isabstract, getmembers

from MDMC.MD.force_fields.ff import ForceField

class ForceFieldFactory(object):

    """
    Provides a factory for creating force fields.  Any facade within
    the force fields folder can be created with a string of the class name, as
    long as it is a subclass of ForceField.
    """

    @staticmethod
    def create_force_field(module_name):
        module = import_module('.' + module_name, __package__)

        classes = getmembers(module, lambda m: (isclass(m)
                                                and not isabstract(m)
                                                and issubclass(m, ForceField)))

        return classes[0][1]()
