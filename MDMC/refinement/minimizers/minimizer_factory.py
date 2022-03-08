
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
    def create_minimizer(module_name, parameter, distribution='uniform',
                         max_parameter_change: float = 0.01, **settings: dict):
        """
        Checks that the module is a supported minimzer and instantiates it as a minimizer.

        Parameters
        ----------
        module_name: str
            The name of the module to be used as the minimizer, e.g. 'MMC'
        parameter: list[str]
            List of parameters to be refined
        distribution: str, optional
            Name of distribution to be used, defaults to 'uniform'
        max_parameter_change: float, optional
            Maximum factor by which a Parameter can change each step of the
            refinement. Defaults to `0.01`
        **settings: dict, optional
            Settings to be passed to the created minimiser, e.g. MC_norm=1.0

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
        # The ** here is dictionary unpacking not passing directly the settings kwargs
        return classes[0][1](parameter, distribution,
                             max_parameter_change , **settings)

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
