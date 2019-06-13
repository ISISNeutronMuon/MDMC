"""Factory class for generating observables"""

from importlib import import_module
from inspect import isclass, isabstract, getmembers

from MDMC.trajectory_analysis.observables.obs import \
    Observable

class ObservableFactory(object):

    """
    Provides a factory for creating Observables.  Any module within the
    observables submodule can be created with a string of the class name, as
    long as it is a subclass of Observable.
    """

    @staticmethod
    def create_observable(module_name):

        """
        Creates an Observable object from a module name

        The Observable object must be the first class in the module

        Parameters
        ----------
        module_name : str
            The name of the module where the Observable is the first class

        Returns
        -------
        Observable
            An Observable object
        """

        module = import_module('.' + module_name, __package__)

        classes = getmembers(module, lambda m: (isclass(m)
                                                and not isabstract(m)
                                                and issubclass(m, Observable)))
        observable = classes[0][1]()
        observable.name = module_name
        return observable
