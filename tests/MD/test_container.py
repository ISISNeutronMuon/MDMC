"""Tests for subclasses of AtomContainer
"""

import pytest

from MDMC.MD.simulation import Universe
from MDMC.MD.structures import Atom, Molecule


ELEMENTS = ['H', 'He', 'O', 'C', 'Dy']
ATOM_COLLECTIONS = ['universe', 'molecule']

@pytest.fixture
def atoms():
    """
    Returns
    -------
    list
        A list of Atoms with the elements defined in ELEMENTS (and in the same
        order)
    """

    return [Atom(element) for element in ELEMENTS]

@pytest.fixture
def universe(atoms):
    """
    Returns
    -------
    Universe
        A Universe containing the list of atoms in the ``atoms`` fixture
    """

    uni = Universe(10., verbose=False)
    for atom in atoms:
        uni.add_structure(atom)
    return uni

@pytest.fixture
def molecule(atoms):
    """
    Returns
    -------
    Molecule
        A Molecule containing the list of atoms in the ``atoms`` fixture
    """

    return Molecule(atoms=atoms)


@pytest.mark.parametrize('atom_collection', ATOM_COLLECTIONS)
def test_atom_collection_index(atom_collection, atoms, request):
    """
    Tests that indexing into a subclass of AtomCollection returns an Atom
    """

    for index, atom in enumerate(atoms):
        assert request.getfixturevalue(atom_collection)[index] == atom


@pytest.mark.parametrize('atom_collection', ATOM_COLLECTIONS)
def test_atom_collection_slice(atom_collection, request):
    """
    Tests that slicing a subclass of AtomCollection returns a list of Atoms
    """

    collection_slice = request.getfixturevalue(atom_collection)[0:4:2]
    assert len(collection_slice) == 2
    assert collection_slice[0].element == 'H'
    assert collection_slice[1].element == 'O'
