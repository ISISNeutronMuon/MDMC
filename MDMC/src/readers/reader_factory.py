"""Factory class for generating readers

AUTHOR :    Thomas Farmer        START DATE :    2018-6-5 17:44:11"""

from importlib import import_module
from inspect import isclass, isabstract, getmembers

from MDMC.src.readers.readers import Reader

class ReaderFactory(object):

    @staticmethod
    def create_reader(module_name):
        module = import_module('.' + module_name, __package__)

        classes = getmembers(module, lambda m: (
                                        isclass(m)
                                        and not isabstract(m)
                                        and issubclass(m, Reader)))

        return classes[0][1]()
