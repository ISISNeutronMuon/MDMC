"""Factory class for generating observables

AUTHOR :    Thomas Farmer        START DATE :    2018-6-5 14:18:49"""

from importlib import import_module
from inspect import isclass, isabstract, getmembers

from MDMC.trajectory_analysis.observables.obs import \
    Observable

class ObservableFactory(object):

    @staticmethod
    def create_observable(module_name):
        module = import_module('.' + module_name, __package__)

        classes = getmembers(module, lambda m: (isclass(m)
                                                and not isabstract(m)
                                                and issubclass(m, Observable)))
        observable = classes[0][1]()
        observable.name = module_name
        return observable
