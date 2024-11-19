"""
Contains functions for applying resolution functions to data, and a factory used to get them.
"""

from .from_file import FileResolution
from .gaussian import GaussianResolution
from .lorentzian import LorentzianResolution
from .null import NullResolution
from .resolution import Resolution
from .resolution_factory import ResolutionFactory

__all__ = [
    "FileResolution",
    "GaussianResolution",
    "LorentzianResolution",
    "NullResolution",
    "Resolution",
    "ResolutionFactory",
]
