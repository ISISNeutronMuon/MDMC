"""Contains the interface to the Atomic Simulation Environment (ASE).

MDMC interfaces to ASE in two ways:
- MDMC Atom objects can be converted to and from ASE Atom objects
- The ASE GUI can be used to plot MDMC Atom and Bond objects

Contents
--------
cif
conversions
viewer
"""

from . import cif
from . import conversions
from . import viewer
