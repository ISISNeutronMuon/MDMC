"""Tests functions related to ase viewers
"""

import pytest

from MDMC.MD.ase import conversions, viewer


class MockAtom:

    def __init__(self, ID):
        self.ID = ID
        self.position = (0., 0., 0.)
        self.element = 'H'
        self.mass = '1.'
        self.charge = 0.
        self.bonded_interactions = []


@pytest.mark.parametrize('n_atoms, atoms_max',
                         [(0, 10),
                          (5, 50),
                          (10, 10)])
def test_limit_atoms(n_atoms, atoms_max):
    """
    Tests that limit_atoms does not change the number of atoms if it is smaller
    than the specified max_atoms
    """

    atoms = conversions.get_ase_atoms([MockAtom(i) for i in range(n_atoms)])
    assert len(atoms) == n_atoms
    assert len(viewer.limit_atoms(atoms, atoms_max)) == n_atoms


@pytest.mark.parametrize('n_atoms, atoms_max',
                         [(100, 10),
                          (1000, 999),
                          (1000, 0)])
def test_limit_atoms_max(n_atoms, atoms_max):
    """
    Tests that limit_atoms limits the number of atoms if it is greater than the
    specified max_atoms, and warns
    """

    atoms = conversions.get_ase_atoms([MockAtom(i) for i in range(n_atoms)])
    assert len(atoms) == n_atoms

    with pytest.warns(UserWarning):
        limited_atoms = viewer.limit_atoms(atoms, atoms_max)

    assert len(limited_atoms) == atoms_max
