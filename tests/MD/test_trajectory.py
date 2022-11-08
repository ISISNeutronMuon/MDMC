"""
Tests for creating Structure, BoundingBox, and Coulombic objects
and setting their attributes.
"""

from copy import deepcopy
from itertools import combinations, permutations

import numpy as np
import pytest
from pytest_cases import parametrize, fixture, fixture_ref, lazy_value

from MDMC.MD.interactions import Bond, BondAngle, Coulombic, Dispersion
from MDMC.MD.simulation import Universe, Shake, PPPM, Simulation
from MDMC.trajectory_analysis.trajectory import Configuration
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory
from MDMC.MD.structures import (Atom, BoundingBox, Molecule,
                                      get_reduced_chemical_formula)
from tests.system_tests.observables.data_manager import trajectory



ATOM_TYPES = [1, 2, 3]
POS_MASS = [((0, 0, 0), 1), ((-1, 2, 1), 2), ((2, 1, -2), 3)]
TEST_CHARGE_1 = 3.14
TEST_CHARGE_2 = -2.71
UNIVERSE_DIMENSIONS = (10., 10., 10.)
NUMBER_OF_STEPS = 5000

@pytest.fixture
def atom():

    """
    Creates an Atom object.
    """

    return Atom('H',cutoff=10.)

@pytest.fixture
def universe():

    """
    Initializes an empty universe object.
    """

    return Universe(UNIVERSE_DIMENSIONS)

@pytest.fixture
def atoms():

    """
    Generates a 3-body atom list with positions and masses defined by a
    global variable.
    """

    return [Atom('H', position=pos, mass=mass) for (pos, mass) in POS_MASS]

@pytest.fixture
def atom_types_universe(atoms, universe):

    """
    Generates a list of atom_types for atoms added to a universe.
    Returns the atom_types and the universe.
    """

    for atom in atoms:
        universe.add_structure(atom)
    return ([atom.atom_type for atom in atoms], universe)

@pytest.fixture
def atom_charge():

    """
    Creates an Atom object initialised with a charge.
    """

    return Atom('H', charge=TEST_CHARGE_1, cutoff=10.)

@pytest.fixture
def water_molecule():

    """
    Returns
    -------
    Molecule
        A water molecule with no interactions (i.e. just atoms defined)
    """

    return Molecule(position=(0, 0, 0),
                    atoms=[Atom('H'),
                           Atom('H', position=(0., 1.63298, 0.)),
                           Atom('O', position=(0., 0.81649, 0.57736))],
                    name='water')

@pytest.fixture(scope='module')
def water_trajectory():
    # Build universe
    # Cubic universe of side:
    # 24.83602653 is 512 water molecules
    universe = Universe(dimensions=24.836)
    H1 = Atom('H')
    H2 = Atom('H', position=(0., 1.63298, 0.))
    O = Atom('O', position=(0., 0.81649, 0.57736))
    H_coulombic = Coulombic(atoms=[H1, H2], cutoff=10.)
    O_coulombic = Coulombic(atoms=O, cutoff=10.)
    water_mol = Molecule(position=(0, 0, 0),
                        velocity=(0, 0, 0),
                        atoms=[H1, H2, O],
                        interactions=[Bond((H1, O), (H2, O), constrained=True),
                                    BondAngle(H1, O, H2, constrained=True)],
                        name='water')
    shake = Shake(1e-4, 100)
    universe.constraint_algorithm = shake
    e_solver = PPPM(accuracy=1e-5)
    universe.electrostatic_solver = e_solver
    universe.fill(water_mol, num_density=0.03356718472021752)
    O_dispersion = Dispersion(universe, (O.atom_type, O.atom_type), cutoff=10.,
                            vdw_tail_correction=True)
    universe.add_force_field('SPCE')

    simulation = Simulation(universe,
                            engine="lammps",
                            time_step=0.5,
                            temperature=280.,
                            traj_step=1)

    # Energy Minimization and equilibration
    simulation.minimize(n_steps=5000)
    simulation.run(n_steps=10000, equilibration=True)
    simulation.run(n_steps=NUMBER_OF_STEPS)
    traj = simulation.trajectory

    yield traj
    simulation.engine.lmp.close()


def test_empty_trajectory():
    traj = CompactTrajectory()
    assert len(traj) == 0

def test_number_of_atoms(water_trajectory):
    traj = water_trajectory
    assert traj.n_atoms == 1536

def test_number_of_elements(water_trajectory):
    traj = water_trajectory
    element_array = np.array(traj.element_list)
    assert np.sum(element_array == 'O') == 512
    assert np.sum(element_array == 'H') == 1024

def test_lammps_trajectory(water_trajectory):
    """
    A LAMMPS 'run 0' command generates a trajectory
    with length 1. For this reason the trajectory
    has to be 1 step longer than the number of steps.
    """
    traj = water_trajectory
    print("First index = ", traj.first_index)
    print("Last index = ", traj.last_index)
    assert len(traj) == NUMBER_OF_STEPS + 1
