"""Tests conversions between equivalent MDMC and ASE objects
"""

import numpy as np
import pytest

import ase

from MDMC.MD.ase import conversions
from MDMC.MD.structural_units import Atom


class MockAtom:

    def __init__(self, ID):
        self.ID = ID


class MockBond:

    def __init__(self, atoms):
        self.atoms = atoms


@pytest.fixture
def bond_atom_IDs():

    """
    Returns
    -------
    list of tuples
        To be used as IDs for atoms in a bond
    """

    return [(1, 2), (3, 4), (5, 6), (1, 6), (3, 6)]


def test_ASEAtoms():

    """
    Tests that an ASEAtoms class is equal to an ase.atoms.Atoms class
    initialized with the same parameters, but also has bonds and IDs attributes.
    """

    formula = 'C8H4O2'
    cell = (1., 2., 3.)
    bonds = [(1, 2)]
    IDs = list(range(0, 14, 1))
    ase_atoms = conversions.ASEAtoms(symbols=formula, cell=cell, bonds=bonds,
                                     IDs=IDs)
    assert ase_atoms == ase.atoms.Atoms(symbols=formula, cell=cell)
    assert ase_atoms.bonds == bonds
    assert ase_atoms.IDs == IDs


def test_ASEAtoms_error():

    """
    Tests that a ValueError is raised if there are not the same number of IDs as
    atoms
    """

    with pytest.raises(ValueError):
        conversions.ASEAtoms(symbols='H5', IDs=range(0, 10, 1))


@pytest.mark.parametrize('position, index, mass, symbol, charge',
                         [((0., 0., 0.), 1, 12., 'Ca', 1.5),
                          ((-5, -10., -15.), 10, 1., 'H', 0.),
                          ((2., 4., 8.), None, 56., 'H', -0.5)])
def test_convert_to_ase_atom(position, index, mass, symbol, charge):

    """
    Tests that an equivalent ase.atom.Atom object is created from an MDMC Atom
    """

    atom = Atom(symbol, position=position, mass=mass, charge=charge)
    ase_atom = conversions.convert_to_ase_atom(atom, index)
    # If an index is not passed, the index should be set to the Atom ID
    index = index if index else atom.ID
    assert ase_atom.symbol == symbol
    assert ase_atom.charge == charge
    assert ase_atom.mass == mass
    assert ase_atom.index == index
    assert all(ase_atom.position == position)


@pytest.mark.parametrize('element, atom_type, name, set_charge',
                         [('H', 1, 'Alcohol OH (UA)', True),
                          ('O', 5, 't-Butanol COH (UA)', False),
                          ('P', None, None, False),
                          ('K', 4, None, True),
                          ('Ca', None, '119', False)])
def test_convert_from_ase_atom(element, atom_type, name, set_charge):

    """
    Tests that an equivalent MDMC Atom object is created from an ase.atom.Atom

    Includes testing that the atom_type, name and charge can be optionally set
    """

    charge = 1.
    ase_atom = ase.atom.Atom(symbol=element, charge=charge)
    atom = conversions.convert_from_ase_atom(ase_atom,
                                             atom_type=atom_type,
                                             name=name,
                                             set_charge=set_charge)
    assert atom.element == element
    assert atom.atom_type == atom_type
    # If a name is not passed, the name should be the symbol
    name = name if name else ase_atom.symbol
    assert atom.name == name
    # If set_charge is False, the charge should be None
    charge = charge if set_charge else None


@pytest.mark.parametrize('IDs, expected',
                         [(range(1, 20, 1),
                           range(0, 19, 1)),
                          (range(1, 50, 2),
                           range(0, 25, 1)),
                          ([7, 4, 19, 55],
                           range(0, 4, 1))])
def test_get_ase_atoms(monkeypatch, IDs, expected):

    """
    Tests that an ASEAtoms object can created from a list of atoms

    Parametrized to test with different IDs, as these require conversion due
    to ASE indexing

    This does not test application of bonds, as these are tested in other
    functions
    """

    atoms = []
    for ID in IDs:
        atom = Atom('H')
        atom.ID = ID
        atoms.append(atom)

    index_conversion = dict(zip(IDs, expected))
    ase_atoms = conversions.get_ase_atoms(atoms)
    for i, atom in enumerate(ase_atoms):
        assert atom.index == index_conversion[ase_atoms.IDs[i]]


def test_convert_bond(bond_atom_IDs):

    """
    Tests that bonds are correctly converted to integers indexing atoms, where
    there is no index conversion
    """

    bond_atoms = map(lambda x: (MockAtom(x[0]), MockAtom(x[1])), bond_atom_IDs)
    mock_bond = MockBond(bond_atoms)

    ase_bond_atoms = conversions.convert_bond(mock_bond)
    assert ase_bond_atoms == bond_atom_IDs


def test_convert_bond_index_conversion(bond_atom_IDs):

    """
    Tests that bonds are correctly converted to integers indexing atoms, where
    there is index conversion
    """

    index_conv = {1:100, 2:200, 3:300, 4:400, 5:500, 6:600}
    bond_atoms = map(lambda x: (MockAtom(x[0]), MockAtom(x[1])), bond_atom_IDs)
    mock_bond = MockBond(bond_atoms)

    ase_bond_atoms = conversions.convert_bond(mock_bond, index_conv=index_conv)
    assert np.all(np.shape(ase_bond_atoms) == np.shape(bond_atom_IDs))
    for ase_pair, ID_pair in zip(ase_bond_atoms, bond_atoms):
        for ase_index, ID in zip(ase_pair, ID_pair):
            assert ase_index == index_conv[ID]
