import os.path

import numpy as np
import pytest

from MDMC.MD import Atom, Bond, BondAngle, Molecule
from MDMC.exporters.configurations.pdb import ProteinDataBankExporter
from tests.test_data.data import _ABS_DIR_PATH

# lammps mark used to ensure test runs in docker container
pytestmark = [pytest.mark.lammps]

@pytest.fixture()
def h2o_molecule():
    """A simple water molecule"""
    # Names added to test exporting functionality - not representative of real FF names
    h_1 = Atom("H", name="20")
    h_2 = Atom('H', position=[0., 1.63298, 0.], name="20")
    o_1 = Atom('O', position=[0., 0.81649, 0.57736], name="21")
    ho_bonds = Bond((h_1, o_1), (h_2, o_1))
    h20_bondangle = BondAngle(atom_tuples=[(h_1, o_1, h_2)])
    h2o_mol = Molecule(atoms=[h_1, h_2, o_1], interactions=[h20_bondangle, ho_bonds])
    return h2o_mol

@pytest.fixture()
def exported_file_path():
    """Returns the file path to the file exported"""
    return os.path.join(_ABS_DIR_PATH, "configurations/example_pdb_export.pdb")

@pytest.fixture()
def simple_exported_system(h2o_molecule, exported_file_path):
    """Exports a simple water molecule to pdb format"""
    pdb_exporter = ProteinDataBankExporter(exported_file_path)
    with pdb_exporter:
        pdb_exporter.write(h2o_molecule)

@pytest.fixture()
def exported_file_text(simple_exported_system, exported_file_path):
    """Gets the lines from the exported file"""
    with open(exported_file_path, "r") as exported_file:
        lines = [line for line in exported_file]
    return lines

def get_atom_lines(file_text):
    """A method to extract ATOM records from the pdb file"""
    atom_lines = []
    for line in file_text:
        if line.startswith("ATOM"):
            atom_lines.append(line)
    return atom_lines

def test_correct_n_atoms_exported(exported_file_text):
    """Tests that the correct number of atoms are exported"""
    atom_lines = get_atom_lines(exported_file_text)
    assert len(atom_lines) == 3

def test_coordinates_are_correct(exported_file_text, h2o_molecule):
    """Tests that the coordinates exported to the file are correct"""
    atom_lines = get_atom_lines(exported_file_text)
    for i, line in enumerate(atom_lines):
        expected_positions = [float(pos.split()[-1]) for pos in (line[30:38], line[38:46], line[46:54])]
        real_positions = h2o_molecule.atoms[i].position.tolist()
        assert np.allclose(expected_positions, real_positions, atol=1e-3)

def test_correct_elements_exported(exported_file_text, h2o_molecule):
    """Tests that the elements exported to the file are correct"""
    atom_lines = get_atom_lines(exported_file_text)
    for i, line in enumerate(atom_lines):
        assert line[76:78].split()[-1] == h2o_molecule.atoms[i].element

def test_correct_names_exported(exported_file_text, h2o_molecule):
    """Tests that the correct name for the atoms are exported (the element)"""
    atom_lines = get_atom_lines(exported_file_text)
    for i, line in enumerate(atom_lines):
        assert line[12:16].split()[-1] == h2o_molecule.atoms[i].element
