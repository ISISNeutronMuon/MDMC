"""Tests for simulation module, both setting up and running a simulation

 AUTHOR :    Thomas Farmer        START DATE :    2018-4-30 13:05:13"""

import pytest
import numpy.testing as npt

import MDMC.src.MD.interaction_functions as ifu
import MDMC.src.MD.structural_units as su
import MDMC.src.MD.simulation as sim

UNIVERSE_DIMS = (10,10,10)
UNIVERSE_SHAPE = sim.Shape.orthorhombic
UNIVERSE_PBC = sim.Boundary.cubic

N_MOLECULES = 1000

@pytest.fixture
def universe():
    return sim.Universe(UNIVERSE_DIMS,UNIVERSE_SHAPE,UNIVERSE_PBC)

@pytest.fixture
def atom():
    return su.Atom('H',mass=1.008)

# TODO: Combine with water box defined in test_structural_units
@pytest.fixture
def water_molecule(atom):
    H1 = atom
    H2 = su.Atom('H',mass=1.008)
    O = su.Atom('O',mass=16.000)
    return su.Molecule(atoms=[H1,H2,O],interactions=[su.Bond(H1,O),su.Bond(H2,O)])

# @opytest.fixture
# def water_universe(water_molecule,universe):
#     water_universe = universe.add(water_molecule,N_MOLECULES)
#     return water_universe


def test_create_universe(universe):
    assert UNIVERSE_SHAPE == universe.shape
    npt.assert_array_equal (UNIVERSE_DIMS,universe.dims)
    assert UNIVERSE_PBC == universe.pbc

def test_create_atom(atom):
    assert 1 == atom.ID
    npt.assert_array_equal((0,0,0),atom.position)
    npt.assert_array_equal((0,0,0),atom.velocity)
    assert 'H' == atom.element
    assert 1.008 == atom.mass
    assert su.NonBonded == type(atom.interaction_set().pop())

def test_atom_list(atom):
    assert atom in atom.atom_list()

def test_add_atom(universe,atom):
    universe.add_structure(atom)
    assert atom.atom_list() == universe.atom_list()
    assert su.NonBonded == type(universe.interaction_set().pop())

def test_add_molecule(universe,water_molecule):
    universe.add_structure(water_molecule)
    assert water_molecule.atom_list() == universe.atom_list()
    for interaction in water_molecule.interaction_set():
        assert ['H','O'] == interaction.sorted_element_list()

def test_spce_water_molecule(universe,water_molecule):
    universe.add_structure(water_molecule)
    universe.add_force_field(SPCE)
    # TODO: Create bonds for a single water molecule and test
    # TODO: Fill universe with molecule and test for bonds

# def test_fill_universe(water_universe):
#     # TODO: Combine with box generated for test_force_fields.py
#     assert 3*N_MOLECULES == len(water_universe.atomList())
#
