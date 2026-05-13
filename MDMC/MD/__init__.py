# MDMC is a package for the optimisation of classical potentials with experimental data
# Copyright (C) 2026 MDMC Developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
from .constraints import Rattle, Shake
from .interaction_functions import (
    Buckingham,
    Coulomb,
    HarmonicPotential,
    InteractionFunction,
    LennardJones,
    NonBonded,
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
from .kspace_solvers import PPPM, Ewald, KSpaceSolver
from .parameters import Parameter, Parameters
from .simulation import (
    ConstraintAlgorithm,
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
for _, name, _ in iter_modules(force_fields.__path__, force_fields.__name__ + "."):
    if name.split(".")[-1] not in ["ff", "force_field_factory"]:
        module = import_module(name)
        cls = getmembers(module, lambda m: isclass(m) and not isabstract(m))[0][1]
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
    "NonBonded",
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
