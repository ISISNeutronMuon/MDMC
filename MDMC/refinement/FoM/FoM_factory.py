"""
Factory class for generating Figure of Merits
And ObservablePair class fro defining the obseravble pairs used to calculate the Figure of Merit
"""
from pathlib import Path


from glob import glob
from importlib import import_module
from inspect import getmembers, isabstract, isclass
from os.path import basename, dirname, isfile, join

from MDMC.refinement.FoM.FoM_abs import FigureOfMerit


class FoMFactory(ModuleFactory[FigureOfMerit]):
    """
    Provides a factory for creating a ``Figure of Merit`` also called FoM.

    Any FoM within the FoM folder can be created with a string of the class name, as
    long as it is a subclass of ``FigureOfMerit``.
    """
    registry: dict[str, FigureOfMerit] = {}
    curr_path = Path(__file__).parent
    curr_pack = __package__
    exclude = (curr_path / "__init__.py", curr_path / "FoM_factory.py")

    @classmethod
    def scan(cls):
        super().scan()

        # Add aliases
        FoMFactory.registry |= {key.lower()
                                .removeprefix("chisquared")
                                .removeprefix("rsquared")
                                .removeprefix("_")
                                .removesuffix("error") : val
                                for key, val in FoMFactory.registry.items()}


FoMFactory.scan()
