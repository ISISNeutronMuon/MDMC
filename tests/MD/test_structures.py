"""
Tests for creating Structure, BoundingBox, and Coulombic objects
and setting their attributes.
"""

from copy import deepcopy
from itertools import combinations, permutations

import numpy as np
import pytest
from pytest_cases import parametrize
import periodictable

from MDMC.MD.interaction_functions import Coulomb
from MDMC.MD.simulation import Universe
from MDMC.MD.structures import (Atom, BoundingBox, Molecule,
                                      get_reduced_chemical_formula)
from MDMC.MD.interactions import Coulombic, Bond, BondAngle
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


def test_charge_creates_coulombic(atom_charge):
    """
    Tests that setting the charge during Atom initialisation
    creates a Coulombic interaction and only a Coulombic
    interaction.
    """

    assert atom_charge.interactions[0].name == 'Coulombic'
    assert len(atom_charge.interactions) == 1


def test_charge_after_init_creates_coulombic(atom):
    """
    Tests that setting the charge after Atom initialisation
    creates a Coulombic interaction and only a Coulombic
    interaction.
    """

    atom.charge = TEST_CHARGE_1
    assert atom.interactions[0].name == 'Coulombic'
    assert len(atom.interactions) == 1


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
def test_atom_charge_cutoff(atom):
    """
    Tests that the cutoff of the Coulombic interaction created when the charge
    of an Atom is set is 10.0, if the Atom did not already possess a Coulombic
    interaction.

    Ignores any warnings thrown.
    """

    atom.charge = TEST_CHARGE_1
    assert atom.interactions[0].cutoff == 10.


@pytest.mark.filterwarnings("ignore:Coulombic interaction")
def test_charge_change_no_coulomb(atom_charge):
    """
    Tests that a charge can be changed after is has already been
    set during Atom initialisation.

    Ignores any warnings thrown.
    """

    atom_charge.charge = TEST_CHARGE_2
    assert atom_charge.charge == TEST_CHARGE_2


@pytest.mark.filterwarnings("ignore:Coulombic interaction")
def test_charge_change_coulomb(atom):
    """
    Tests that a charge can be changed after it has already
    been set during initialisation of a Coulombic interaction.

    Ignores any warnings thrown.
    """

    Coulombic(atoms=atom, charge=TEST_CHARGE_1)
    atom.charge = TEST_CHARGE_2
    assert atom.charge == TEST_CHARGE_2


@pytest.mark.filterwarnings("ignore:Coulombic interaction")
def test_charge_when_none(atom):
    """
    Tests that setting the charge of an Atom of charge None that
    has a Coulombic interaction creates an interaction function.
    """

    Coulombic(atoms=atom)
    atom.charge = TEST_CHARGE_1
    assert atom.interactions[0].function.name == 'Coulomb'
    assert isinstance(atom.interactions[0].function, Coulomb)


def test_charge_get_when_none(atom):
    """
    Tests that getting the charge of an atom initialised without specifying
    a charge returns a charge of None.
    """

    assert atom.charge is None


@pytest.mark.filterwarnings("ignore:Coulombic interaction")
def test_charge_set_zero(atom):
    """
    Tests that when the charge of an initialised atom is set to zero
    that the charge returns zero and that a Coulombic interaction has
    been created.
    """

    atom.charge = 0
    assert atom.charge == 0.0
    assert atom.interactions[0].name == 'Coulombic'


def test_charge_getter_checks(atom_charge):
    """
    Tests that an error is raised when trying to retrieve the charge of
    an atom that has 2 Coulombic interactions.
    """

    # A second atom has to be added as otherwise the Coulombic interaction is
    # not unique, and is therefore not added to atom_charge
    Coulombic(atoms=[atom_charge, deepcopy(atom_charge)])
    with pytest.raises(ValueError):
        atom_charge.charge


def test_charge_no_cutoff():
    """
    Tests that not supplying a cutoff value for a charged atom
    raises a warning and uses the default cutoff of 10.
    """

    with pytest.warns(UserWarning):
        atom = Atom('H', charge=5.)

    assert atom.cutoff == 10.

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


def test_init_coulombic_atoms_no_universe(atoms):
    """
    Tests that a Coulombic interaction can be initialised by passing
    atoms as a parameter.
    """

    coul = Coulombic(atoms=atoms, charge=TEST_CHARGE_1)
    assert all(coul.atoms) == all(atoms)
    assert coul.parameters['charge'].value == TEST_CHARGE_1


def test_init_coulombic_atoms_added_to_universe(atoms, universe):
    """
    Tests that a Coulombic interaction can be initialised by passing
    atoms and universe as parameters, where the Atoms have been
    added to the universe.
    """

    for atom in atoms:
        universe.add_structure(atom)
    coul = Coulombic(universe, atoms=atoms, charge=TEST_CHARGE_1)
    assert isinstance(coul.universe, Universe)


def test_init_coulombic_atoms_not_added_to_universe(atoms, universe):
    """
    Tests that a Coulombic interacion can be initialised by passing
    atoms and universe as parameters, where the Atoms have not been
    added to the universe.

    Tests that the universe property of the Coulombic object is None.
    """

    assert (Coulombic(universe, atoms=atoms, charge=TEST_CHARGE_1).universe
            is None)


def test_init_coulombic_atom_types_universe(atom_types_universe):
    """
    Tests that a Coulombic interaction can be initialized by passing
    atom_types and universe as parameters, where the Atoms for which
    the atom_types are specified have been added to the universe.
    """

    coul = Coulombic(atom_types_universe[1], atom_types=atom_types_universe[0],
                     charge=TEST_CHARGE_1)
    assert isinstance(coul.universe, Universe)
    assert all(coul.atom_types) == all(atom_types_universe[0])
    assert coul.parameters['charge'].value == TEST_CHARGE_1


def test_init_coulombic_error_atom_types_no_universe():
    """
    Tests that an error is thrown when atom_types is passed as a
    parameter without passing a universe object.
    """

    with pytest.raises(TypeError):
        Coulombic(atom_types=[1, 2, 3], charge=TEST_CHARGE_1)


def test_init_coulombic_error_atoms_and_atom_types(atoms,
                                                   atom_types_universe):
    """
    Tests that an error is thrown when both atoms and atom_types are
    passed as parameters when initialising a Coulombic interaction.
    """

    with pytest.raises(TypeError):
        Coulombic(atom_types_universe[1], atoms=atoms,
                  atom_types=atom_types_universe[0], charge=TEST_CHARGE_1)


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


def test_neutral_atom_has_no_charge(atom, atom_charge):
    """
    Tests that when an Atom is added with no charge,
    it is not given a charge parameter, and if an
    atom is added *with* charge, it is.
    """

    assert len(atom.interactions) == 0
    assert len(atom_charge.interactions) == 1


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

def test_deepcopy_copies_existing_interactions(water_molecule, universe):
    """Testing that the CompositeStructure.__deepcopy__ method correctly copies
     the interactions of the CompositeStructure that is being copied. """
    universe = universe
    universe.add_structure(water_molecule)
    universe.add_force_field('SPCE')
    # there should be 4 parameters for SPCE water:
    #  2 for H-O Bonds with HarmonicPotential: equilibrium bond length, bond strength
    #  2 for H-O-H BondAngle with HarmonicPotential: equilibrium bond angle, bond strength
    assert 4 == len(universe.parameters)
    water_copy = water_molecule.copy([1.,1.,1.])
    # there should be 3 interactions in the copied molecule: 
    # 2 H-O Bonds, 1 H-O-H BondAngle
    assert 3 == len(water_copy.interactions)
    universe.add_structure(water_copy)
    # the number of parameters in the Universe should be unchanged when the copied molecule is added
    assert 4 == len(universe.parameters)
