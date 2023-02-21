"""Tests for reading pdb files"""
import pytest
from numpy.testing import assert_allclose
from MDMC.MD import Atom, Molecule, BondAngle
from MDMC.readers.configurations import read
from tests.test_data import data

@pytest.fixture
def water_molecule():
    """A fixture of an SPCE water molecule"""
    H1 = Atom('H')
    H2 = Atom('H', position=[0., 1.63298, 0.])
    O = Atom('O', position=[0., 0.81649, 0.57736])
    HOH_angle = BondAngle(H1, O, H2)
    return Molecule(atoms=[H1, O, H2], interactions=[HOH_angle])

def test_single_structure_read(water_molecule):
    atoms = read(data.CONFIG_DATA['pdb_water'])
    assert type(atoms) == list
    water_atoms = water_molecule.atoms
    for i in range(len(atoms)):
        assert atoms[i].name == water_atoms[i].name
        assert_allclose(atoms[i].position, water_atoms[i].position)

def test_whole_system_read():
    pass
