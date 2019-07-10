"""
Tests for creating structural unit objects and setting
their attributes.

AUTHOR :    Joe Abbott        START DATE :    09/07/2019, 10:30:25
"""

import pytest

import MDMC.MD.structural_units as su


TEST_CHARGE = 3.14
TEST_CHARGE_2 = -2.71


@pytest.fixture
def atom():
    """
    Creates an Atom object.
    """

    return su.Atom('H')


@pytest.fixture
def atom_charge():
    """
    Creates an Atom object with a charge.
    """

    return su.Atom('H', charge=TEST_CHARGE)


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
    initialised with a charge value.
    """

    atom_Coul_charge = atom
    su.Coulombic(atoms=atom_Coul_charge, charge=TEST_CHARGE)
    return atom_Coul_charge


@pytest.mark.filterwarnings("ignore:Coulombic interaction")
def test_charge(atom):
    """
    Tests that the charge attribute of the Atom object can
    be set during Atom initialisation.

    Ignores any warnings thrown.
    """

    assert su.Atom('O', charge=TEST_CHARGE).charge == TEST_CHARGE


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
    H1.charge = TEST_CHARGE
    assert H1.charge == TEST_CHARGE


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


def test_charge_when_None(atom_Coulombic):
    """
    Tests that setting the charge of an Atom of charge None that
    has a Coulomb interaction creates an interaction function.

    """

    H1 = atom_Coulombic
    H1.charge = TEST_CHARGE
    assert H1.interactions[0].function.name == 'Coulomb'
