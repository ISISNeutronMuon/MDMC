"""
Contains functions for applying resolution functions to data, and a factory used to get them.
"""

from .resolution import Resolution
from .resolution_factory import ResolutionFactory
from .gaussian import GaussianResolution
from .lorentzian import LorentzianResolution
from .null import NullResolution
from .from_file import FileResolution
