
"""
Factory class for generating Figure of Merits
And ObservablePair class fro defining the obseravble pairs used to calculate the Figure of Merit
"""


from glob import glob
from importlib import import_module
from inspect import isclass, isabstract, getmembers
from os.path import basename, dirname, join, isfile

from MDMC.refinement.FoM.FoM_abs import FigureOfMerit


class FoMFactory:

    """
    Provides a factory for creating a ``Figure of Merit`` also called FoM.  Any FoM within
    the FoM folder can be created with a string of the class name, as
    long as it is a subclass of ``FigureOfMerit``.
    """

    @staticmethod
    def create_FoM(module_name, obs_pairs, norm: str = 'data_points', n_parameters: int = None):
        """
        Returns
        -------
        FigureOfMerit
            A ``Figure of Merit`` specified by ``module_name``
        """

        try:
            module = import_module(
                '.ChiSquared_' + module_name+'error', __package__)
        except ImportError:
            try:
                module = import_module('.RSquared_'+ module_name+'error', __package__)
            except ImportError:
                try:
                    module = import_module('.' + module_name, __package__)
                except ImportError as error:
                    raise ValueError(
                        f'{module_name} is not a supported Figure of Merits') from error

        classes = getmembers(module, lambda m: (isclass(m)
                                                and not isabstract(m)
                                                and issubclass(m, FigureOfMerit)))
        return classes[0][1](obs_pairs, norm, n_parameters)

    @staticmethod
    def get_FoM_names():
        """
        Get the names of available Figure of Merits(FoM)

        Requires all ``Figure of Merits`` derived classes to be in modules of the same
        name

        Returns
        -------
        list
            A `list` of `str` with the names of the available ``Figure of Merits``
            objects
        """

        FoM_names = []
        for full_module_name in glob(join(dirname(__file__), "*.py")):
            if isfile(full_module_name) and full_module_name != __file__:
                module_name = basename(full_module_name)
                if not module_name.startswith('_') and module_name != 'FoM_abs.py':
                    FoM_names.append(module_name.replace('.py', ''))

        return FoM_names
