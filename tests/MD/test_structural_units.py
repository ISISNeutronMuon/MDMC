"""
Tests for creating structural unit objects and setting
their attributes.
"""

import pytest

from MDMC.MD.interaction_functions import Coulomb
from MDMC.MD.simulation import Shape, Universe
from MDMC.MD.structural_units import Atom, Coulombic


UNIVERSE_DIMS = (10., 10., 10.)
UNIVERSE_SHAPE = Shape.cubic

TEST_CHARGE_1 = 3.14
TEST_CHARGE_2 = -2.71
ATOM_TYPES = [1, 2, 3]


@pytest.fixture
def atom():

    """
    Creates an Atom object.
    """

    return Atom('H')

@pytest.fixture
def atoms(atom):

    """
    Generates a list containing an Atom object.
    """

    return [atom]

@pytest.fixture
def universe():

    """
    Initializes an empty universe object.
    """

    return Universe(UNIVERSE_DIMS, UNIVERSE_SHAPE)

@pytest.fixture
def atom_types_universe(atom, universe):

    """
    Generates a list of atom_types for atoms added to a universe.
    Returns the atom_types and the universe.
    """

    universe.add_structural_unit(atom)
    return ([atom.atom_type], universe)

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

    Coulombic(atoms=atom_charge)
    with pytest.raises(ValueError):
        atom_charge.charge

def test_init_coulombic_atoms_no_universe(atoms):

    """
    Tests that a Coulombic interaction can be initialised by passing
    atoms as a parameter.
    """

    coul = Coulombic(atoms=atoms, charge=TEST_CHARGE_1)
    assert coul.atoms[0] == atoms[0]
    assert coul.params[0].value == TEST_CHARGE_1

def test_init_coulombic_atoms_added_to_universe(atoms, universe):

    """
    Tests that a Coulombic interaction can be initialised by passing
    atoms and universe as parameters, where the Atoms have been
    added to the universe.
    """

    universe.add_structural_unit(atoms[0])
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
    assert coul.atom_types[0] == atom_types_universe[0][0]
    assert coul.params[0].value == TEST_CHARGE_1

def test_init_coulombic_error_atom_types_no_universe(atom_types_universe):

    """
    Tests that an error is thrown when atom_types is passed as a
    parameter without passing a universe object.

    Tests that an error is thrown when atom_types and universe are
    passed as parameters, but the the Atoms for which the atom_types
    are specified have not been added to the universe.
    """

    with pytest.raises(TypeError):
        Coulombic(atom_types=atom_types_universe[0],
                  charge=TEST_CHARGE_1)

def test_init_coulombic_error_atoms_and_atom_types(atoms, atom_types_universe):

    """
    Tests that an error is thrown when both atoms and atom_types are
    passed as parameters when initialising a Coulombic interaction.
    """

    with pytest.raises(TypeError):
        Coulombic(atoms=atoms, atom_types=atom_types_universe[0],
                  charge=TEST_CHARGE_1)
