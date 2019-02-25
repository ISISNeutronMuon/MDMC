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
EQUILIBRIUM_STEPS = 1000
MD_STEPS = 2000

# Each EXPECTED dictionary contains all of the required properties as keys. The
# corresponding values are a tuple of (mean value, standard deviation),
# where both the mean value and the standard deviation have been calculated from
# 10 repeats (with different random velocity seeds) of an external LAMMPS
# simulation with the same simulation parameters.
#
# The NVE temperature differs from the set value due to the effects of SHAKE
NVE_EXPECTED = {'Atoms':(N_MOLECULES*3, 0), 'Bonds':(N_MOLECULES*2, 0),
                'Angles':(N_MOLECULES, 0), 'KinEng':(1523.88, 6.1),
                'PotEng':(-1234.32, 4.8), 'Temp':(1186.15, 4.7),
                'Press':(18105.41, 134.8), 'Volume':(DIMENSION**3, 0),
                'E_bond':(0, 0), 'E_angle':(0, 0), 'E_vdwl':(390.61, 3.5),
                'E_coul':(11601.41, 7.0), 'E_long':(-13226.34, 0.50),
                'Nbuild':(922.24, 3.2), 'Ndanger':(0, 0)}

