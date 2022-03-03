
"""Factory class for generating minimizers"""

from glob import glob
from importlib import import_module
from inspect import isclass, isabstract, getmembers
from os.path import basename, dirname, join, isfile

from MDMC.refinement.minimizers.minimizer_abs import Minimizer


class MinimizerFactory:

    """
    Provides a factory for creating a ``Minimizer``.  Any minimizer within
    the minimizers folder can be created with a string of the class name, as
    long as it is a subclass of ``Minimizer``.
    """

    @staticmethod
    def create_minimizer(module_name, MC_norm, parameter, distribution='uniform',
                         max_parameter_change: float = 0.01):
        """
        Returns
        -------
        Minimizer
            A ``Minimizer`` specified by ``module_name``
        """

        try:
            module = import_module('.' + module_name, __package__)
        except ImportError as error:
            raise ValueError(
                f'{module_name} is not a supported minimizer') from error
        classes = getmembers(module, lambda m: (isclass(m)
                                                and not isabstract(m)
                                                and issubclass(m, Minimizer)))
        return classes[0][1](MC_norm, parameter, distribution,
                             max_parameter_change)

    @staticmethod
    def get_minimizer_names():
        """
        Get the names of available minimizer

        Requires all ``Minimizer`` derived classes to be in modules of the same
        name

        Returns
        -------
        list
            A `list` of `str` with the names of the available ``Minimizer``
            objects
        """

        minimizer_names = []
        for full_module_name in glob(join(dirname(__file__), "*.py")):
            if isfile(full_module_name) and full_module_name != __file__:
                module_name = basename(full_module_name)
                if not module_name.startswith('_') and module_name != 'minimizer_abs.py':
                    minimizer_names.append(module_name.replace('.py', ''))

        return minimizer_names
