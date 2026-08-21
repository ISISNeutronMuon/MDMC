"""
Tests for creating Structure and BoundingBox objects
and setting their attributes.
"""

from copy import deepcopy
from itertools import combinations, permutations

import numpy as np
import pytest
from pytest_cases import parametrize
import periodictable

from MDMC.MD.simulation import Universe
from MDMC.MD.structures import (Atom, BoundingBox, Molecule,
                                      get_reduced_chemical_formula)
from MDMC.MD.interactions import Bond, BondAngle
from MDMC.MD.structures import Atom

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

    return Universe(UNIVERSE_DIMENSIONS, verbose=False)

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
    H1 = Atom('H')
    H2 = Atom('H', position=(0., 1.63298, 0.))
    O = Atom('O', position=(0., 0.81649, 0.57736))
    return Molecule(position=(0, 0, 0),
                    atoms=[H1, H2, O],
                    interactions=[Bond((H1, O), (H2, O), constrained=True),
                                  BondAngle(H1, O, H2, constrained=True)],
                    name='water')


@pytest.mark.filterwarnings("ignore:Coulombic interaction")
def test_charge():
    """
    Tests that the charge attribute of the Atom object can
    be set during Atom initialisation.

    Ignores any warnings thrown.
    """

    assert Atom('O', charge=TEST_CHARGE_1, cutoff=10.).charge == TEST_CHARGE_1


@pytest.mark.filterwarnings("ignore:Coulombic interaction")
def test_charge_after_init(atom):
    """
    Tests that the charge of an atom can be set after
    initialisation, without there existing a Coulombic
    interaction.

    Ignores any warnings thrown.
    """

    atom.charge = TEST_CHARGE_1
    assert atom.charge == TEST_CHARGE_1


@pytest.mark.filterwarnings("ignore:Coulombic interaction")
def test_charge_change_no_coulomb(atom_charge):
    """
    Tests that a charge can be changed after is has already been
    set during Atom initialisation.

    Ignores any warnings thrown.
    """

    atom_charge.charge = TEST_CHARGE_2
    assert atom_charge.charge == TEST_CHARGE_2


def test_bounding_box_empty_raises_value_error():
    """
    Tests that passing an empty atom list raises a value Error
    """

    with pytest.raises(ValueError):
        BoundingBox(atoms=[])

@parametrize('atoms_size', [1, 2, 3])
def test_bounding_box_min(atoms, atoms_size):
    """
    Tests that the min property of the BoundingBox class returns the correct
    value for a 1, 2, and 3-bodied atom list.
    """

    body_list = atoms[:atoms_size]
    mn = atoms[0].position
    for atom in body_list:
        mn = np.minimum(mn, atom.position)
    bb_mn = BoundingBox(body_list).min
    np.testing.assert_array_equal(bb_mn, mn)



@parametrize('atoms_size', [1, 2, 3])
def test_bounding_box_max(atoms, atoms_size):
    """
    Tests that the max property of the BoundingBox class returns the correct
    value for a 1, 2, and 3-bodied atom list.
    """

    body_list = atoms[:atoms_size]
    mx = atoms[0].position
    for atom in body_list:
        mx = np.maximum(mx, atom.position)
    bb_mx = BoundingBox(body_list).max
    np.testing.assert_array_equal(bb_mx, mx)


@parametrize('atoms_size', [1, 2, 3])
def test_bounding_box_volume(atoms, atoms_size):
    """
    Tests that the correct volume is returned for the bounding box of a
    1, 2, and 3-bodied atom list.
    """
    
    body_list = atoms[:atoms_size]
    bb = BoundingBox(body_list)
    assert bb.volume == abs(np.prod(bb.max - bb.min))


@parametrize('atoms_size', [1, 2, 3])
def test_molecule_mass(atoms, atoms_size):
    """
    Tests that the mass property returns the expected result for 1, 2,
    and 3-bodied Molecule objects.
    """

    body_list = atoms[:atoms_size]
    mol = Molecule(atoms=body_list)
    exp_mass = 0.
    for atom in body_list:
        exp_mass += atom.mass
    assert mol.mass == exp_mass


@pytest.mark.parametrize('position', [(0., 0., 0.),
                                      (5., 5., 5.),
                                      (1., 2., 3.)])
def test_molecule_rotation_preserves_CoM(position, water_molecule):
    """
    Tests that the molecular center of mass remains constant when
    Molecule.rotate is called
    """

    water_molecule.position = position
    for x, y, z in permutations([0., 45., 90.]):
        water_molecule.rotate(x=x, y=y, z=z)
        assert all(water_molecule.position == position)


@pytest.mark.parametrize('angles', [(90., 0., 0.),
                                    (0., 90., 0.),
                                    (0., 0., 90.),
                                    (45., 45., -45.)])
def test_molecule_rotation_preserves_distances(angles, water_molecule):
    """
    Tests that the distances between atoms are preserved when Molecule.rotate is
    called
    """

    def get_separations(atoms):

        # rounding is just to avoid floating point errors
        return [round(np.linalg.norm(atom1.position - atom2.position), 5)
                for atom1, atom2 in combinations(atoms, 2)]

    initial_separations = get_separations(water_molecule.atoms)
    water_molecule.rotate(x=angles[0], y=angles[1], z=angles[2])
    final_separations = get_separations(water_molecule.atoms)
    assert initial_separations == final_separations


@pytest.mark.parametrize('angles, expected',
                         [((90., 0., 0.), [0., 0.51275077, -0.81649]),
                          ((0., 90., 0.), [-0.51275077, -0.81649, 0.]),
                          ((0., 0., 90.), [0.81649, 0., -0.51275077]),
                          ((90., 90., 90.), [-0.51275077, -0.81649, 0.]),
                          ((360., 360., 360.), [0., -0.81649, -0.51275077])])
def test_molecule_rotation(angles, expected, water_molecule):
    """
    Tests that the rotation results in the expected position for first H atom

    H atom starts at: [0., -0.81649 , -0.51275077]
    """

    water_molecule.rotate(x=angles[0], y=angles[1], z=angles[2])
    assert np.allclose(water_molecule.atoms[0].position, expected,
                       atol=1e-5)


@pytest.mark.parametrize('symbols, factor, formula, system',
                         [(['C'] * 4 + ['H'] * 16,
                           4,
                           'CH4',
                           'Hill'),
                          (['C'] * 4 + ['H'] * 16,
                           None,
                           'CH4',
                           'Hill'),
                          (['C'] * 24 + ['H'] * 27 + ['N'] * 3 + ['O'] * 6,
                           None,
                           'C8H9NO2',
                           'Hill'),
                          (['C'] * 24 + ['H'] * 27 + ['N'] * 3 + ['O'] * 6,
                           None,
                           'C8H9NO2',
                           None),
                          (['O'] * 6 + ['N'] * 3 + ['H'] * 27 + ['C'] * 24,
                           None,
                           'O2NH9C8',
                           None),
                          (['C'] * 24 + ['H'] * 56 + ['N'] * 8 + ['O'] * 8,
                           4,
                           'C6H14N2O2',
                           'Hill')])
def test_get_reduced_chemical_formula_error(symbols, factor, formula, system):
    """
    Tests that get_reduced_chemical_formula returns the reduced chemical formula
    for the symbols, based on the factor. Includes tests both with and without a
    passed factor, with different orderings of the symbols, and with both `Hill`
    system and no system.
    """

    assert get_reduced_chemical_formula(symbols, factor, system) == formula


@pytest.mark.parametrize('element', ['H', 'O','Pb', 'Ca'])
def test_periodictable_elements(element):
    """
    Tests that different elements and checmical symbols created by MDMC match (in properties) those created straight from the periodictable module.
    """
    test_atom = Atom(element, name='test atom')
    actual_atom = periodictable.elements.symbol(element)
    
    assert type(test_atom.element) is periodictable.core.Element
    assert str(test_atom.element) == element == actual_atom.symbol


@pytest.mark.parametrize('atom_type, element, isotope_num'
                         , [('He[3]', 'He',3),('U[235]', 'U',235),
                            ('He[3]', 'He', 3), ('K[40]', 'K', 40),
                        ('He[4]', 'He', 4)])
def test_periodictable_isotopes_and_abundances(atom_type,element,isotope_num):
    """
    Tests that different isotopes of elements created by MDMC match (in properties)
    those created straight from the periodictable module. This also tests the notation
    of specifying mass number for standard elements (such as He[4]).
    """
    test_atom = Atom(atom_type, name='test atom')
        
    actual_atom = periodictable.elements.symbol(element)[isotope_num]
    
    assert type(test_atom.element) is periodictable.core.Isotope
    assert str(test_atom.element) == '%d-%s' % (isotope_num, element)
    assert test_atom.element.abundance == actual_atom.abundance


@pytest.mark.parametrize('element', ['X', 'Fo', 'He[5]', 'Si[20]'])
def test_periodictable_invalid_element_or_isotope(element):
    """
    This tests that using a wrong elemental symbol or specifying a non-existent isotope raises exceptions with atom creation.
    """
    with pytest.raises(Exception):
        Atom(element, 'test atom')


@pytest.mark.parametrize('atom_type, element, isotope_num'
                         , [('He[3]', 'He',3),('U[235]', 'U',235),
                            ('He[3]', 'He', 3), ('K[40]', 'K', 40),
                        ('He[4]', 'He', 4), ('O', None, None), ('Ca', None, None)])
def test_periodictable_properties(atom_type, element, isotope_num):
    """
    Tests that some important MDMC atoms properties are identical to those of an atom made straight from periodictable.
    """
    test_atom = Atom(atom_type, name='test atom')
    
    if isotope_num is not None:
        actual_atom = periodictable.elements.symbol(element)[isotope_num]
    else:
        actual_atom = periodictable.elements.symbol(atom_type)
    
    assert test_atom.mass == actual_atom.mass
    assert test_atom.element.number == actual_atom.number
    assert test_atom.element.neutron == actual_atom.neutron
    assert test_atom.element.density == actual_atom.density
