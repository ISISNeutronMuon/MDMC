"""Factory class for generating MD engine facades

AUTHOR :    Thomas Farmer        START DATE :    2018-5-23 13:22:22"""

from importlib import import_module
from inspect import isclass, isabstract, getmembers

from MDMC.src.MD.engine_facades.facade import MDEngine

class MDEngineFacadeFactory(object):

    """
    Provides a factory for creating facades to MD engines.  Any facade within
    the engine_facades folder can be created with a string of the class name, as
    long as it is a subclass of MDEngine.
    """

    @staticmethod
    def create_facade(module_name):
        module = import_module('.' + module_name, __package__)

        classes = getmembers(module, lambda m: (
                                        isclass(m)
                                        and not isabstract(m)
                                        and issubclass(m, MDEngine)))

        return classes[0][1]()
