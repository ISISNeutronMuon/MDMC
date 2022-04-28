"""Classes and functions for setting up and running a molecular dynamics
simulation

Contents
--------
ase
engine_facades
force_fields
interaction_functions
solvents
simulation
structures
interactions
"""

from importlib import import_module
from pkgutil import iter_modules
from inspect import isclass, isabstract, getmembers

from . import ase
from . import engine_facades
from . import force_fields
from . import solvents
from .interaction_functions import *
from .simulation import *
from .structures import *
from .interactions import *

# Get the class of each force field
for _, name, _ in iter_modules(force_fields.__path__,
                               force_fields.__name__ + '.'):
    if name.split('.')[-1] not in ['ff', 'force_field_factory']:
        module = import_module(name)
        cls = getmembers(module, lambda m: (isclass(m)
                                            and not isabstract(m)))[0][1]
        globals()[cls.__name__] = cls
