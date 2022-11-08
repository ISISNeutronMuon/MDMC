"""
Tests for creating Structure, BoundingBox, and Coulombic objects
and setting their attributes.
"""

from copy import deepcopy
from itertools import combinations, permutations

import numpy as np
import pytest
from pytest_cases import parametrize, fixture, fixture_ref, lazy_value

from MDMC.MD.simulation import Universe
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


def test_empty_trajectory():
    traj = CompactTrajectory()
    assert len(traj) == 0

def test_create_trajectory(universe, water_molecule):
    traj = CompactTrajectory(n_steps=10, n_atoms=3, universe = universe)
    traj.setDimensions(UNIVERSE_DIMENSIONS,0)
    conf = Configuration(water_molecule)
    for step_num, time_val in enumerate(np.arange(0.0,10.0,1.0)):
        traj.writeOneStep(step_num=step_num, time=time_val,
                          positions=conf.atom_positions)
    assert traj.n_atoms == 3
    assert len(traj) == 10

def test_populate_trajectory():
    traj = CompactTrajectory()
    assert len(traj) == 0
