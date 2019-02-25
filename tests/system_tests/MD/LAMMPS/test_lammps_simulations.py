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

NVT_EXPECTED = {'Atoms':(N_MOLECULES*3, 0), 'Bonds':(N_MOLECULES*2, 0),
                'Angles':(N_MOLECULES, 0), 'KinEng':(385.73, 1.2), 'PotEng':(-2408.00, 5.3),
                'Temp':(300.24, 0.91), 'Press':(159.47, 132.7),
                'Volume':(DIMENSION**3, 0), 'E_bond':(0, 0), 'E_angle':(0, 0),
                'E_vdwl':(457.34, 5.4), 'E_coul':(10388.75, 10.1), 'E_long':(-13255.09, 0.09),
                'Nbuild':(384.39, 2.4), 'Ndanger':(0, 0)}

NPT_EXPECTED = {'Atoms':(N_MOLECULES*3, 0), 'Bonds':(N_MOLECULES*2, 0),
                'Angles':(N_MOLECULES, 0), 'KinEng':(385.33, 1.2), 'PotEng':(-2407.60, 6.3),
                'Temp':(299.93, 0.90), 'Press':(-11.83, 34.4),
                'Volume':(6470.59, 35.2), 'E_bond':(0, 0), 'E_angle':(0, 0),
                'E_vdwl':(454.16, 4.1), 'E_coul':(10374.46, 22.4), 'E_long':(-13236.22, 24.5),
                'Nbuild':(417.04, 2.9), 'Ndanger':(0, 0)}


