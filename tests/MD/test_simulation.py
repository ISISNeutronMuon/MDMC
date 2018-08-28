"""Tests for setting up a simulation

 AUTHOR :    Thomas Farmer        START DATE :    2018-4-30 13:05:13"""

from collections import Counter
from copy import deepcopy

import numpy as np
import numpy.testing as npt
import pytest

import MDMC.MD.force_fields as ff
import MDMC.MD.simulation as sim
import MDMC.MD.structural_units as su


UNIVERSE_DIMS = (10., 10., 10.)
UNIVERSE_SHAPE = sim.Shape.cubic

H1_POSITION = (0., 0., 0.)
H2_POSITION = (0.151390, 0., 0.)
O_POSITION = (0.075695, 0., 0.058588)
H_MASS = 1.008
O_MASS = 16.000
WATER_POSITION = (1., 2., 3.)
WATER_NUM_DENSITY = 0.0333679


@pytest.fixture
def universe():

    return sim.Universe(UNIVERSE_DIMS, UNIVERSE_SHAPE)

@pytest.fixture
def atom():

    return su.Atom('H', mass=H_MASS)

# TODO: Combine with water box defined in test_structural_units
@pytest.fixture
def water_molecule(atom):

    H1 = atom
    H2 = su.Atom('H', position=H2_POSITION, mass=H_MASS)
    O = su.Atom('O', position=O_POSITION, mass=O_MASS)
    water_molecule = su.Molecule(position=WATER_POSITION, atoms=[H1, H2, O],
                                 interactions=[su.Bond(H1, O), su.Bond(H2, O),
                                               su.Dispersion(O)], name='water')
    water_molecule.add_interaction(su.BondAngle(atoms=[H1, O, H2]))
    return water_molecule

@pytest.fixture
def water_SPCE_universe(water_molecule, universe):

    water_universe = deepcopy(universe)
    water_universe.fill(water_molecule, force_field=ff.SPCE,
                        num_density=WATER_NUM_DENSITY)
    return water_universe


def test_create_universe(universe):

    assert UNIVERSE_SHAPE == universe.shape
    npt.assert_array_equal(UNIVERSE_DIMS, universe.dims)

def test_create_atom(atom):

    npt.assert_array_equal((0., 0., 0.), atom.position)
    npt.assert_array_equal((0., 0., 0.), atom.velocity)
    assert atom.element == 'H'
    assert atom.mass == 1.008
    assert su.Coulombic == type(atom.interactions.pop())

def test_atom_list(atom):

    assert atom in atom.atom_list

def test_add_atom(universe, atom):

    universe.add_structural_unit(atom)
    assert atom.atom_list == universe.atom_list
    assert su.Coulombic == type(universe.interactions.pop())

def test_add_molecule(universe, water_molecule):

    universe.add_structural_unit(water_molecule)
    assert water_molecule.position.all() == np.array(WATER_POSITION).all()
    assert sorted(water_molecule.atom_list) == sorted(universe.atom_list)

    water_CoM = ((np.array(H1_POSITION) * H_MASS)
                 + (np.array(H2_POSITION) * H_MASS)
                 + (np.array(O_POSITION) * O_MASS)) / (H_MASS * 2 + O_MASS)

    CoM_frame_correction = water_CoM + WATER_POSITION
    atom_pos_water_CoM_frame = [H1_POSITION - CoM_frame_correction,
                                H2_POSITION - CoM_frame_correction,
                                O_POSITION - CoM_frame_correction]

    for i in range(len(atom_pos_water_CoM_frame)):
        assert atom_pos_water_CoM_frame[i].all() ==\
                                    water_molecule.atom_list[i].position.all()

    # Test interactions have expected element lists - 1 bond angle, 2 H-O bonds,
    # 1 dispersive on O, 1 Coulombic on O, and 2 Coulombic on H
    interaction_elements = []
    for interaction in water_molecule.interactions:
        interaction_elements.append(interaction.sorted_element_list())
    assert sorted([['H', 'H', 'O'], ['H', 'O'], ['H', 'O'], ['O'], ['O'], ['H'],
                   ['H']]) == sorted(interaction_elements)

def test_spce_water_molecule(universe, water_molecule):

    universe.add_structural_unit(water_molecule)
    universe.add_force_field(ff.SPCE)

    functions = [inter.function for inter in universe.interactions]
    function_names = [function.__class__.__name__ for function in functions]

    # Test interaction functions
    assert Counter(['Coulomb', 'Coulomb', 'Coulomb', 'LennardJones',
                    'HarmonicPotential', 'HarmonicPotential',
                    'HarmonicPotential']) == Counter(function_names)

    # A list of dictionaries with each dictionary containing a Parameter type
    # and the correspoding Parameter value
    params = [{p.name:p.value} for function in functions
              for p in function.params]

    # Test interaction parameters
    SPCEparams = [{'charge':-0.8476}, {'charge':0.4238}, {'charge':0.4238},
                  {'sigma':3.166, 'eta':0.6502},
                  {'equilibrium_state':1.000, 'potential_strength':4637.},
                  {'equilibrium_state':1.000, 'potential_strength':4637.},
                  {'equilibrium_state':109.47, 'potential_strength':383.}]
    for param in params:
        assert param in SPCEparams
        # Remove the instance so that multiple identical instances are tested
        SPCEparams.remove(param)

def test_spce_water_box(water_SPCE_universe):

    n_molecules_xyz = np.array(UNIVERSE_DIMS) * WATER_NUM_DENSITY**(1./3.)
    n_molecules = np.prod(n_molecules_xyz.astype(int))

    assert int(n_molecules) == \
        len(water_SPCE_universe.configuration.molecule_list)

    # TODO: Test for correct positions
    # water_positions = sorted([list(structural_unit.position) for structural_unit
    #                                 in water_SPCE_universe.configuration])
    # intermol_dist = np.array(UNIVERSE_DIMS) / int(n_molecules**(1./3.))
    # calc_positions = []
    # for x in np.arange(0, UNIVERSE_DIMS[0], intermol_dist[0]):
    #     for y in np.arange(0, UNIVERSE_DIMS[1], intermol_dist[1]):
    #         for z in np.arange(0, UNIVERSE_DIMS[2], intermol_dist[2]):
    #             calc_positions.append([x, y, z])
    # assert sorted(calc_positions) == water_positions
