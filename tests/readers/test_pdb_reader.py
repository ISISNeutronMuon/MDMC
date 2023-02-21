"""Tests for reading pdb files"""
from MDMC.MD import Atom, Molecule, BondAngle
from MDMC.readers.configurations import read
from tests.test_data import data

def water_molecule():
    """A fixture of an SPCE water molecule"""
    H1 = Atom('H')
    H2 = Atom('H', position=[0., 1.63298, 0.])
    O = Atom('O', position=[0., 0.81649, 0.57736])
    HOH_angle = BondAngle(H1, O, H2)
    return Molecule(atoms=[H1, O, H2], interactions=[HOH_angle])

def test_single_structure_read():
    molecules = read(data.CONFIG_DATA['pdb_water'])
    assert len(molecules) == 1

def test_whole_system_read():
    pass
