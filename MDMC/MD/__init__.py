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
from inspect import getmembers, isabstract, isclass
from pkgutil import iter_modules

from . import ase, engine_facades, force_fields, solvents
from .interaction_functions import (
    Buckingham,
    Coulomb,
    HarmonicPotential,
    InteractionFunction,
    LennardJones,
    Periodic,
    inter_func_decorator,
)
from .interactions import (
    Bond,
    BondAngle,
    BondedInteraction,
    ConstrainableMixin,
    Coulombic,
    DihedralAngle,
    Dispersion,
    Interaction,
    NonBondedInteraction,
)
from .parameters import Parameter, Parameters
from .simulation import (
    PPPM,
    ConstraintAlgorithm,
    Ewald,
    KSpaceSolver,
    Rattle,
    Shake,
    Simulation,
    Universe,
)
from .structures import (
    Atom,
    BoundingBox,
    CompositeStructure,
    Molecule,
    Structure,
    filter_atoms,
    filter_atoms_element,
    get_reduced_chemical_formula,
)

# Get the class of each force field
for _, name, _ in iter_modules(force_fields.__path__,
                               force_fields.__name__ + '.'):
    if name.split('.')[-1] not in ['ff', 'force_field_factory']:
        module = import_module(name)
        cls = getmembers(module, lambda m: (isclass(m)
                                            and not isabstract(m)))[0][1]
        globals()[cls.__name__] = cls

__all__ = [
    "ase",
    "engine_facades",
    "force_fields",
    "solvents",
    "Buckingham",
    "Coulomb",
    "HarmonicPotential",
    "InteractionFunction",
    "LennardJones",
    "Periodic",
    "inter_func_decorator",
    "Bond",
    "BondAngle",
    "BondedInteraction",
    "ConstrainableMixin",
    "Coulombic",
    "DihedralAngle",
    "Dispersion",
    "Interaction",
    "NonBondedInteraction",
    "Universe",
    "KSpaceSolver",
    "Ewald",
    "PPPM",
    "ConstraintAlgorithm",
    "Shake",
    "Rattle",
    "Simulation",
    "Structure",
    "CompositeStructure",
    "Atom",
    "Molecule",
    "BoundingBox",
    "filter_atoms",
    "filter_atoms_element",
    "get_reduced_chemical_formula",
    "Parameters",
    "Parameter",
]
