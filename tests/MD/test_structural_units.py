"""
Tests for creating structural unit objects and setting
their attributes.

AUTHOR :    Joe Abbott        START DATE :    09/07/2019, 10:30:25
"""

import pytest

import MDMC.MD.simulation as sim
import MDMC.MD.structural_units as su


UNIVERSE_DIMS = (10., 10., 10.)
UNIVERSE_SHAPE = sim.Shape.cubic

TEST_CHARGE_1 = 3.14
TEST_CHARGE_2 = -2.71
ATOM_TYPES = [1, 2, 3]


@pytest.fixture
def atom():
    """
    Creates an Atom object.
    """

    return su.Atom('H')


@pytest.fixture
def atoms(atom):
    """
    Generates a list containing an Atom object.
    """

    return [atom]


@pytest.fixture
def universe():
    """
    Initializes a universe object.
    """

    return sim.Universe(UNIVERSE_DIMS, UNIVERSE_SHAPE)


@pytest.fixture
def atom_types_universe(atom, universe):
    """
    Generates a list of atom_types for atoms added to a universe.
    Returns the atom_types and the universe.
    """
    univ = universe
    H1 = atom
    univ.add_structural_unit(H1)
    return [H1.atom_type], univ


@pytest.fixture
def atom_charge():
    """
    Creates an Atom object initialised with a charge.
    """

    return su.Atom('H', charge=TEST_CHARGE_1)


@pytest.fixture
def atom_Coulombic(atom):
    """
    Creates an Atom object with an Coulombic interaction.
    """
    atom_Coul = atom
    su.Coulombic(atoms=atom_Coul)
    return atom_Coul


@pytest.fixture
def atom_Coulombic_charge(atom):
    """
    Creates an Atom object, with a Coulombic interaction
    initialised with a chIs a test charge value.
    """

    atom_Coul_charge = atom
    su.Coulombic(atoms=atom_Coul_charge, charge=TEST_CHARGE_1)
    return atom_Coul_charge


@pytest.mark.filterwarnings("ignore:Coulombic interaction")
def test_charge(atom):
    """
    Tests that the charge attribute of the Atom object can
    be set during Atom initialisation.

    Ignores any warnings thrown.
    """

    assert su.Atom('O', charge=TEST_CHARGE_1).charge == TEST_CHARGE_1


def test_charge_creates_Coulombic(atom_charge):
    """
    Tests that setting the charge during Atom initialisation
    creates a Coulombic interaction and only a Coulombic
    interaction.
    """

    H1 = atom_charge
    assert H1.interactions[0].name == 'Coulombic'
    assert len(H1.interactions) == 1


def test_charge_after_init_creates_Coulombic(atom_charge):
    """
    Tests that setting the charge after Atom initialisation
    creates a Coulombic interaction and only a Coulombic
    interaction.
    """

    H1 = atom_charge
    assert H1.interactions[0].name == 'Coulombic'
    assert len(H1.interactions) == 1


@pytest.mark.filterwarnings("ignore:Coulombic interaction")
def test_charge_after_init(atom):
    """
    Tests that the charge of an atom can be set after
    initialisation, without there existing a Coulombic
    interaction.

    Ignores any warnings thrown.
    """

    H1 = atom
    H1.charge = TEST_CHARGE_1
    assert H1.charge == TEST_CHARGE_1


@pytest.mark.filterwarnings("ignore:Coulombic interaction")
def test_charge_change_no_Coulomb(atom_charge):
    """
    Tests that a charge can be changed after is has already been
    set during Atom initialisation.

    Ignores any warnings thrown.
    """

    H1 = atom_charge
    H1.charge = TEST_CHARGE_2
    assert H1.charge == TEST_CHARGE_2


@pytest.mark.filterwarnings("ignore:Coulombic interaction")
def test_charge_change_Coulomb(atom_Coulombic_charge):
    """
    Tests that a charge can be changed after it has already
    been set during initialisation of a Coulombic interaction.

    Ignores any warnings thrown.
    """

    H1 = atom_Coulombic_charge
    H1.charge = TEST_CHARGE_2
    assert H1.charge == TEST_CHARGE_2


def test_charge_set_warning(atom):
    """
    Tests that a warning is raised when the charge of an atom is
    set without there being a pre-existing a Coulombic interaction.
    """

    H1 = atom
    value = -0.6
    with pytest.warns(Warning) as record:
        H1.charge = value
        if not record:
            pytest.fail('Expected a warning!')


@pytest.mark.filterwarnings("ignore:Coulombic interaction")
def test_charge_when_None(atom_Coulombic):
    """
    Tests that setting the charge of an Atom of charge None that
    has a Coulomb interaction creates an interaction function.
    """

    H1 = atom_Coulombic
    H1.charge = TEST_CHARGE_1
    assert H1.interactions[0].function.name == 'Coulomb'


def test_init_Coulombic_atoms_no_universe(atoms):
    """
    Tests that a Coulombic interaction can be initialised by passing
    atoms as a parameter.
    """

    H1_atoms = atoms
    coul = su.Coulombic(atoms=H1_atoms, charge=TEST_CHARGE_1)
    assert coul.atoms[0] == H1_atoms[0]
    assert coul.params[0].value == TEST_CHARGE_1


def test_init_Coulombic_atoms_added_to_universe(atoms, universe):
    """
    Tests that a Coulombic interaction can be initialised by passing
    atoms and universe as parameters, where the Atoms have been
    added to the universe.
    """

    H1_atoms = atoms
    univ = universe
    univ.add_structural_unit(H1_atoms[0])
    coul = su.Coulombic(universe, atoms=H1_atoms, charge=TEST_CHARGE_1)
    assert isinstance(coul.universe, sim.Universe)


def test_init_Coulombic_atoms_not_added_to_universe(atoms, universe):
    """
    Tests that a Coulombic interacion can be initialised by passing
    atoms and universe as parameters, where the Atoms have not been
    added to the universe.

    Tests that the universe property of the Coulombic object is None.
    """

    H1_atoms = atoms
    univ = universe
    coul = su.Coulombic(universe, atoms=H1_atoms, charge=TEST_CHARGE_1)
    assert coul.universe == None


def test_init_Coulombic_atom_types_universe(atom_types_universe):
    """
    Tests that a Coulombic interaction can be initialized by passing
    atom_types and universe as parameters, where the Atoms for which
    the atom_types are specified have been added to the universe.
    """

    H1_types, univ = atom_types_universe
    coul = su.Coulombic(univ, atom_types=H1_types, charge=TEST_CHARGE_1)
    assert isinstance(coul.universe, sim.Universe)
    assert coul.atom_types[0] == H1_types[0]
    assert coul.params[0].value == TEST_CHARGE_1


def test_init_Coulombic_error_atom_types_no_universe(atom_types_universe):
    """
    Tests that an error is thrown when atom_types is passed as a
    parameter without passing a universe object.

    Tests that an error is thrown when atom_types and universe are
    passed as parameters, but the the Atoms for which the atom_types
    are specified have not been added to the universe.
    """

    H1_types, _ = atom_types_universe
    try:
        coul = su.Coulombic(atom_types=H1_types, charge=TEST_CHARGE_1)
    except TypeError:
        pass
    else:
        pytest.fail('Expected a TypeError')


def test_init_Coulombic_error_atoms_and_atom_types(atoms, atom_types_universe):
    """
    Tests that an error is thrown when both atoms and atom_types are
    passed as parameters when initialising a Coulombic interaction.
    """

    H1_atoms = atoms
    H1_types, _ = atom_types_universe
    try:
        coul = su.Coulombic(atoms=H1_atoms, atom_types=H1_types,
                            charge=TEST_CHARGE_1)
    except TypeError:
        pass
    else:
        pytest.fail('Expected a TypeError')
