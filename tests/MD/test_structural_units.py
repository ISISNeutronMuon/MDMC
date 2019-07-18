"""
Tests for creating structural unit objects and setting
their attributes.
"""

import numpy as np
import pytest

from MDMC.MD.interaction_functions import Coulomb
from MDMC.MD.structural_units import Atom, BoundingBox, Coulombic, Molecule

TEST_CHARGE = 3.14
TEST_CHARGE_2 = -2.71
POS_MASS = [((0, 0, 0), 1), ((-1, 2, 1), 2), ((2, 1, -2), 3)]


@pytest.fixture
def atom():

    """
    Creates an Atom object.
    """

    return Atom('H')


@pytest.fixture
def atom_charge():

    """
    Initialises an Atom object with a charge.
    """

    return Atom('H', charge=TEST_CHARGE)

@pytest.fixture
def atom_list():

    """
    Generates a 3-body atom list with positions and masses defined by a
    global variable.
    """

    return [Atom('H', position=pos, mass=mass) for (pos, mass) in POS_MASS]


@pytest.mark.filterwarnings("ignore:Coulombic interaction")
def test_charge():

    """
    Tests that the charge attribute of the Atom object can
    be set during Atom initialisation.

    Ignores any warnings thrown.
    """

    assert Atom('O', charge=TEST_CHARGE).charge == TEST_CHARGE


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

    atom.charge = TEST_CHARGE
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

    atom.charge = TEST_CHARGE
    assert atom.charge == TEST_CHARGE


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

    Coulombic(atoms=atom, charge=TEST_CHARGE)
    atom.charge = TEST_CHARGE_2
    assert atom.charge == TEST_CHARGE_2


def test_charge_set_warning(atom):

    """
    Tests that a warning is raised when the charge of an atom is
    set without there being a pre-existing a Coulombic interaction.
    """

    with pytest.warns(UserWarning):
        atom.charge = TEST_CHARGE


@pytest.mark.filterwarnings("ignore:Coulombic interaction")
def test_charge_when_none(atom):

    """
    Tests that setting the charge of an Atom of charge None that
    has a Coulomb interaction creates an interaction function.
    """

    Coulombic(atoms=atom)
    atom.charge = TEST_CHARGE
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
    Tests that an error is raised when trying to retrieve the charge of an
    atom that has 2 Coulombic interactions.
    """

    Coulombic(atoms=atom_charge)
    with pytest.raises(ValueError):
        atom_charge.charge


@pytest.mark.parametrize('atom_list', [atom_list()[:i] for i in range(1, 4)])
def test_bounding_box_min(atom_list):

    """
    Tests that the min property of the BoundingBox class returns the correct
    value for a 1, 2, and 3-bodied Molecule.
    """

    mn = atom_list[0].position
    for atom in atom_list:
        mn = np.minimum(mn, atom.position)
    bb_mn = BoundingBox(atom_list).min
    np.testing.assert_array_equal(bb_mn, mn)


@pytest.mark.parametrize('atom_list', [atom_list()[:i] for i in range(1, 4)])
def test_bounding_box_min(atom_list):

    """
    Tests that the min property of the BoundingBox class returns the correct
    value for a 1, 2, and 3-bodied Molecule.
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
    1, 2, and 3-bodied molecule.
    """

    mol = Molecule(atoms=atom_list)
    bb = mol.bounding_box
    assert bb.volume == abs(np.prod(bb.max - bb.min))
