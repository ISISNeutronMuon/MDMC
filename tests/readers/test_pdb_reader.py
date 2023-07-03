"""Tests for reading pdb files"""
import os

import pytest
from numpy.testing import assert_allclose

from MDMC.MD import Atom, Molecule, Bond
from MDMC.readers.configurations import pdb
from tests.test_data.data import _ABS_DIR_PATH


@pytest.fixture
def pdb_ethanol_data_path():
    """
    Returns the path to the directory for the example pdb configuration file for ethanol
    """
    return os.path.join(_ABS_DIR_PATH, "configurations/ethanol.pdb")

@pytest.fixture()
def ethanol_molecule():
    """Returns a `Molecule` object equivalent to that which the pdb file should have read in"""
    c1 = Atom(element="C", position=(-4.914, 1.802, 0.137))
    c2 = Atom(element="C", position=(-3.588, 1.243, -0.406))
    h1 = Atom(element="H", position=(-4.728, 2.627, 0.828))
    h2 = Atom(element="H", position=(-5.532, 2.173, -0.684))
    h3 = Atom(element="H", position=(-5.472, 1.025, 0.664))
    h4 = Atom(element="H", position=(-3.792, 0.424, -1.101))
    h5 = Atom(element="H", position=(-3.046, 2.032, -0.935))
    o1 = Atom(element="O", position=(-2.794, 0.764, 0.680))
    h6 = Atom(element="H", position=(-1.959, 0.413, 0.321))
    ch_bonds = Bond((c1, h1), (c1, h2), (c1, h3), (c2, h4), (c2, h5))
    cc_bond = Bond((c1, c2))
    co_bond = Bond((c2, o1))
    oh_bond = Bond((o1, h6))
    return Molecule(atoms=[c1,c2,h1,h2,h3,h4,h5,o1,h6], interactions=[ch_bonds,cc_bond,co_bond,oh_bond])

def test_single_structure_read(pdb_ethanol_data_path, ethanol_molecule):
    """Tests that the pdb reader will read a single structrue"""
    reader = pdb.ProteinDataBankReader(pdb_ethanol_data_path)
    with reader:
        reader.parse()

    # Assert atoms are the same
    assert type(reader.atoms) == list
    assert len(reader.atoms) == 9
    water_atoms = ethanol_molecule.atoms
    for i in range(len(reader.atoms)):
        assert reader.atoms[i].element == water_atoms[i].element
        assert_allclose(reader.atoms[i].position, water_atoms[i].position)

    # Ensure bonds are correct
    read_ethanol = Molecule(atoms=reader.atoms, interactions=reader.bonds)
    assert len(read_ethanol.bonded_interactions) == 8
