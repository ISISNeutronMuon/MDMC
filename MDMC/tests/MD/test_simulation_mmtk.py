"""Tests for setting up and running a simulation using the MMTK engine

AUTHOR :    Thomas Farmer        START DATE :    2018-5-25 10:57:25"""

import pytest
import numpy as np
import numpy.testing as npt

from MDMC.tests.MD.test_simulation import universe, atom, water_molecule, \
    water_SPCE_universe, UNIVERSE_DIMS

import MDMC.src.MD.simulation as sim
import MDMC.src.MD.engine_facades.mmtk as mmtk

from MMTK.Dynamics import VelocityVerletIntegrator

TIME_STEP = 1.
TEMPERATURE = 310
N_STEPS = 2
INTEGRATOR = 'velocity_verlet'
LJ_OPTIONS = 1.2
ES_OPTIONS = {'method':'ewald'}

@pytest.fixture
def water_MMTK_NVE(water_SPCE_universe):
    return sim.NVESimulation(water_SPCE_universe, engine="mmtk",
        time_step = TIME_STEP, temperature = TEMPERATURE,
        integrator = INTEGRATOR, lj_options = LJ_OPTIONS,
        es_options = ES_OPTIONS)

# TODO: When support for more MD engines is added, parameterize these tests
def test_MMTK_universe_setup(water_MMTK_NVE):

    """
    Test the following equivalencies:

    Universe dimensions
    Number of atoms
    Molecule positions
    Force field type
    Force field parameters
    """

    MMTK_universe = water_MMTK_NVE.engine.universe
    MDMC_universe = MMTK_universe.MDMC_universe
    npt.assert_array_equal(MMTK_universe.data,np.array(UNIVERSE_DIMS))
    assert len(MMTK_universe.atomList()) == len(MDMC_universe.atom_list)

    MMTK_molecule_positions = [np.array(mol.position()) for mol
        in MMTK_universe.objectList()]
    MDMC_molecule_positions = [mol.position for mol
        in MDMC_universe.molecule_list]
    for i in range(len(MMTK_molecule_positions)):
        npt.assert_allclose(MMTK_molecule_positions[i],
            MDMC_molecule_positions[i],
            atol = 1e-5)

    assert type(MMTK_universe.forcefield()) == \
        mmtk.UNIVERSE_FF[type(MDMC_universe.force_fields)]

    # TODO: Test for force field parameters

def test_MMTK_simulation_setup(water_MMTK_NVE):

    """
    MMTK generates a Boltzmann dsitribution of velocities based on the
    temperature, which can actually deviate from the specified temperature
    significantly for small systems.  Therefore the test temperatures are
    rounded to the nearest 100 K.
    The factor of 1000 accounts for the fact that in MDMC timesteps are
    specified in fs, whereas they are specified in ps in MMTK.
    """

    assert type(water_MMTK_NVE.engine.integrator) ==  VelocityVerletIntegrator
    assert water_MMTK_NVE.engine.integrator.options['delta_t'] == \
        TIME_STEP / 1000.
    assert round(water_MMTK_NVE.engine.universe.temperature(),-2) == \
        round(TEMPERATURE,-2)

def test_MMTK_simulation_run(water_MMTK_NVE):

    """
    Test the following equivalencies:

    Total simulation time
    Temperature
    Density
    """

    water_MMTK_NVE.run(N_STEPS)
    npt.assert_allclose((TIME_STEP / 1000) * N_STEPS,
        water_MMTK_NVE.engine.trajectory.time[-1],
        rtol = 1e-5)
    assert round(water_MMTK_NVE.engine.universe.temperature(),-2) == \
        round(TEMPERATURE,-2)
    # TODO: Add density test

def test_MMTK_trajectory_convert(water_MMTK_NVE):

    """
    Run an MMTK simulation and determines the trajectory.  The trajectory is
    converted to an MDMC trajectory.

    Test for same atomic positions in each configuration of the trajectory,
    accounting for the difference in coordinate systems.
    """

    water_MMTK_NVE.run(N_STEPS)
    MDMC_traj = water_MMTK_NVE.trajectory
