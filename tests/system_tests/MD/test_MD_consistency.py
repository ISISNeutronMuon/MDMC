"""System tests for MD engines consistency

Tests that running the same starting simulation setup both through MDMC and
using the MD engine produces the same results, using several different
properties to confirm this.  THIS UNIT TEST TAKES A LONG
TIME TO RUN - IT IS PROBABLY PREFERABLE TO ONLY RUN IT OVERNIGHT DURING
AUTOMATED TESTING.

CURRENTLY ONLY TESTS SIMULATIONS USING MMTK ENGINE, SPCE FORCEFIELD AND NPT
ENSEMBLE

AUTHOR :    Thomas Farmer        START DATE :    05/11/2018, 10:47:14"""

import pytest

from MDMC.MD.simulation import Universe, Shape, Simulation
import MDMC.MD.structural_units as su

"""
Define a starting configuration for a simulation
Define some properties to test
Calculate the expected statistical variation in those properties
Compare the properties with those calculated externally to MDMC
"""

# Universe parameters
SIDE = 18.6270199
SHAPE = Shape.orthorhombic
NUM_DENSITY = 0.0335
FF = 'SPCE'

# Simulation parameters
ENGINE = 'mmtk'
d_t = 1.5
TEMP = 263.
PRESS = 1.
INT = 'velocity_verlet'
LJ_CUTOFF = 12
ES_OPTIONS = 'ewald'
MINIMIZER = 'steepest_descent'
TRAJ_STEP = 2250
THREADS = 4

# Testing parameters
N_REPEATS = 3


@pytest.fixture
def universe():

    """
    Creates a water filled universe
    """

    uni = Universe(dimensions=SIDE, shape=SHAPE)
    H1 = su.Atom('H')
    H2 = su.Atom('H', position=(1.51390, 0., 0.))
    O = su.Atom('O', position=(0.75695, 0., 0.58588))
    water_mol = su.Molecule(position=(0, 0, 0),
                            velocity=(0, 0, 0),
                            atoms=[H1, H2, O],
                            interactions=[su.Bond(H1, O),
                                          su.Bond(H2, O),
                                          su.Dispersion(O),
                                          su.BondAngle(atoms=[H1, O, H2])],
                            name='water')

    uni.fill(water_mol, force_field=FF, num_density=NUM_DENSITY)

    return uni


@pytest.fixture
def MDMC_properties(universe):

    """
    Runs a simulation using a universe and returns properties from this
    simulation
    """

    raise NotImplementedError


@pytest.fixture
def MD_engine_properties():

    raise NotImplementedError


def test_consistency(MDMC_properties, MD_engine_properties):

    """
    Test that simulations runs in MDMC have the same properties (within
    statistical uncertainty) as those calculated from an external run
    """

    pass
