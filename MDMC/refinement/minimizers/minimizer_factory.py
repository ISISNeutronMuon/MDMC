"""Factory class for generating minimizers"""
from glob import glob
from importlib import import_module
from inspect import getmembers, isabstract, isclass
from os.path import basename, dirname, isfile, join
from pathlib import Path
from typing import TYPE_CHECKING

from MDMC.common.factory import ModuleFactory
from MDMC.refinement.minimizers.minimizer_abs import Minimizer

if TYPE_CHECKING:
    from MDMC.control import Control

class MinimizerFactory(ModuleFactory[Minimizer]):
    """
    Provides a factory for creating a ``Minimizer``.

    Any minimizer within
    the minimizers folder can be created with a string of the class name, as
    long as it is a subclass of ``Minimizer``.
    """
    registry: dict[str, Minimizer] = {}
    curr_path = Path(__file__).parent
    curr_pack = __package__
    exclude = (curr_path / "__init__.py", curr_path / "minimizer_factory.py")

MinimizerFactory.scan()
