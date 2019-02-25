"""System tests for LAMMPS MD simulations

Compares the thermodynamic and simulation properties calculated from the MDMC
run using LAMMPS with the same properties calculated from an equivalent LAMMPS
setup run externally. This occurs for NVE, NVT and NPT ensembles.  The
calculations of the properties in both cases are performed by LAMMPS, the only
difference is whether the LAMMPS simulation was run through MDMC.

AUTHOR :    Thomas Farmer        START DATE :    22/02/2019, 13:50:29"""

import numpy as np
import pytest

from MDMC.MD.simulation import Universe, Simulation, Shake, PPPM
from MDMC.MD.structural_units import Atom, Bond, BondAngle, Coulombic, \
    Dispersion, Molecule

N_MOLECULES = 216
DIMENSION = 18.63
TEMPERATURE = 300.
# Number of steps between logging of thermo_style variables
THERMO_STEPS = 100
MD_STEPS = 1000

