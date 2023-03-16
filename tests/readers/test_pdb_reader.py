"""Tests for reading pdb files"""
import pytest
from numpy.testing import assert_allclose
from pytest_cases import fixture

from MDMC.MD import Atom, Molecule, Bond
from MDMC.readers.configurations import pdb
import tests.test_data.data as test_data
from MDMC.readers.configurations.packmol_pdb import PackmolPDBReader


@fixture
def packmol_data_path():
    """
    Returns the path to the directory for the packmol configuration files
    (in the docker container)
    """
    return test_data._ABS_DIR_PATH + test_data._PACKMOL_PATH

@pytest.fixture
def water_molecule():
    """A fixture of an SPCE water molecule"""
    H1 = Atom('H', position=[0., 0., 0.])
    H2 = Atom('H', position=[0., 1.632, 0.])
    O = Atom('O', position=[0., 0.816, 0.577])
    Bond1 = Bond(H1, O)
    Bond2 = Bond(H2, O)
    return Molecule(atoms=[H1, H2, O], interactions=[Bond1, Bond2])

def test_single_structure_read(water_molecule, packmol_data_path):
    reader = pdb.ProteinDataBankReader(
        packmol_data_path + "/water.pdb")
    reader.__enter__()
    reader.parse()

    # Assert atoms are the same
    assert type(reader.atoms) == list
    assert len(reader.atoms) == 3
    water_atoms = water_molecule.atoms
    for i in range(len(reader.atoms)):
        assert reader.atoms[i].name == water_atoms[i].name
        assert_allclose(reader.atoms[i].position, water_atoms[i].position)

    # Ensure bonds are correct
    read_water = Molecule(atoms=reader.atoms, interactions=reader.bonds)
    assert len(read_water.bonded_interactions) == 2

def test_whole_system_read(packmol_data_path):
    reader = PackmolPDBReader(packmol_data_path + "/bilayer.pdb")
    reader.__enter__()
    reader.parse()

    # Assert atoms are correct
    len(reader.atoms) == 8000

    # Test bonds are correct
    len(reader.bonds) == 7000

    # Assert molecules are correct
    len(reader.molecules) == 1100
