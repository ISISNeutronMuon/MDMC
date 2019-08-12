"""
Tests for creating StructuralUnit, BoundingBox, and Coulombic objects
and setting their attributes.
"""

from copy import deepcopy

import numpy as np
import pytest

from MDMC.MD.interaction_functions import Coulomb
from MDMC.MD.simulation import Shape, Universe
from MDMC.MD.structural_units import Atom, BoundingBox, Coulombic, Molecule


ATOM_TYPES = [1, 2, 3]
POS_MASS = [((0, 0, 0), 1), ((-1, 2, 1), 2), ((2, 1, -2), 3)]
TEST_CHARGE_1 = 3.14
TEST_CHARGE_2 = -2.71
UNIVERSE_DIMS = (10., 10., 10.)
UNIVERSE_SHAPE = Shape.cubic


@pytest.fixture
def atom():

    """
    Creates an Atom object.
    """

    return Atom('H')

@pytest.fixture
def universe():

    """
    Initializes an empty universe object.
    """

    return Universe(UNIVERSE_DIMS, UNIVERSE_SHAPE)

@pytest.fixture
def atom_list():

    """
    Generates a 3-body atom list with positions and masses defined by a
    global variable.
    """

    return [Atom('H', position=pos, mass=mass) for (pos, mass) in POS_MASS]

@pytest.fixture
def atom_types_universe(atom_list, universe):

    """
    Generates a list of atom_types for atoms added to a universe.
    Returns the atom_types and the universe.
    """

    for atom in atom_list:
        universe.add_structural_unit(atom)
    return ([atom.atom_type for atom in atom_list], universe)

@pytest.fixture
def atom_charge():

    """
    Creates an Atom object initialised with a charge.
    """

    return Atom('H', charge=TEST_CHARGE_1)


@pytest.mark.filterwarnings("ignore:Coulombic interaction")
def test_charge():

    """
    Tests that the charge attribute of the Atom object can
    be set during Atom initialisation.

    Ignores any warnings thrown.
    """

    assert Atom('O', charge=TEST_CHARGE_1).charge == TEST_CHARGE_1


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


def test_charge_set_warning(atom):

    """
    Tests that a warning is raised when the charge of an atom is
    set without there being a pre-existing a Coulombic interaction.
    """

    with pytest.warns(UserWarning):
        atom.charge = TEST_CHARGE_1


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


@pytest.mark.parametrize('atom_list', [atom_list()[:i] for i in range(1, 4)])
def test_bounding_box_min(atom_list):

    """
    Tests that the min property of the BoundingBox class returns the correct
    value for a 1, 2, and 3-bodied atom list.
    """

    mn = atom_list[0].position
    for atom in atom_list:
        mn = np.minimum(mn, atom.position)
    bb_mn = BoundingBox(atom_list).min
    np.testing.assert_array_equal(bb_mn, mn)


@pytest.mark.parametrize('atom_list', [atom_list()[:i] for i in range(1, 4)])
def test_bounding_box_max(atom_list):

    """
    Tests that the max property of the BoundingBox class returns the correct
    value for a 1, 2, and 3-bodied atom list.
    """

    mx = atom_list[0].position
    for atom in atom_list:
        mx = np.maximum(mx, atom.position)
    bb_mx = BoundingBox(atom_list).max
    np.testing.assert_array_equal(bb_mx, mx)


@pytest.mark.parametrize('atom_list', [atom_list()[:i] for i in range(1, 4)])
def test_bounding_box_volume(atom_list):

    """
    Tests that the correct volume is returned for the bounding box of a
    1, 2, and 3-bodied atom list.
    """

    bb = BoundingBox(atom_list)
    assert bb.volume == abs(np.prod(bb.max - bb.min))


def test_init_coulombic_atoms_no_universe(atom_list):

    """
    Tests that a Coulombic interaction can be initialised by passing
    atoms as a parameter.
    """

    coul = Coulombic(atoms=atom_list, charge=TEST_CHARGE_1)
    assert all(coul.atoms) == all(atom_list)
    assert coul.params[0].value == TEST_CHARGE_1


def test_init_coulombic_atoms_added_to_universe(atom_list, universe):

    """
    Tests that a Coulombic interaction can be initialised by passing
    atoms and universe as parameters, where the Atoms have been
    added to the universe.
    """

    for atom in atom_list:
        universe.add_structural_unit(atom)
    coul = Coulombic(universe, atoms=atom_list, charge=TEST_CHARGE_1)
    assert isinstance(coul.universe, Universe)


def test_init_coulombic_atoms_not_added_to_universe(atom_list, universe):

    """
    Tests that a Coulombic interacion can be initialised by passing
    atoms and universe as parameters, where the Atoms have not been
    added to the universe.

    Tests that the universe property of the Coulombic object is None.
    """

    assert (Coulombic(universe, atoms=atom_list, charge=TEST_CHARGE_1).universe
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
    assert coul.params[0].value == TEST_CHARGE_1


def test_init_coulombic_error_atom_types_no_universe():

    """
    Tests that an error is thrown when atom_types is passed as a
    parameter without passing a universe object.
    """

    with pytest.raises(TypeError):
        Coulombic(atom_types=[1, 2, 3], charge=TEST_CHARGE_1)


def test_init_coulombic_error_atoms_and_atom_types(atom_list,
                                                   atom_types_universe):

    """
    Tests that an error is thrown when both atoms and atom_types are
    passed as parameters when initialising a Coulombic interaction.
    """

    with pytest.raises(TypeError):
        Coulombic(atom_types_universe[1], atoms=atom_list,
                  atom_types=atom_types_universe[0], charge=TEST_CHARGE_1)


@pytest.mark.parametrize('atom_list', [atom_list()[:i] for i in range(1, 4)])
def test_molecule_mass(atom_list):

    """
    Tests that the mass property returns the expected result for 1, 2,
    and 3-bodied Molecule objects.
    """

    mol = Molecule(atoms=atom_list)
    exp_mass = 0.
    for atom in atom_list:
        exp_mass += atom.mass
    assert mol.mass == exp_mass
