import os.path
import re

import pytest

from MDMC.MD import Atom, Bond, BondAngle, Molecule
from MDMC.MD.packmol.packmol_setup import PackmolSetup
from MDMC.exporters.configurations.packmol_input import PackmolInputExporter
from tests.test_data.data import _ABS_DIR_PATH
@pytest.fixture()
def h2o_molecule():
    """A simple water molecule"""
    h_1 = Atom("H")
    h_2 = Atom('H', position=[0., 1.63298, 0.])
    o_1 = Atom('O', position=[0., 0.81649, 0.57736])
    ho_bonds = Bond((h_1, o_1), (h_2, o_1))
    h20_bondangle = BondAngle(atom_tuples=[(h_1, o_1, h_2)])
    h2o_mol = Molecule(atoms=[h_1, h_2, o_1], interactions=[h20_bondangle, ho_bonds], name="water")
    return h2o_mol

@pytest.fixture()
def simple_packmol_input_system(h2o_molecule):
    """A simple packmol setup"""
    setup = PackmolSetup()
    setup.add_cube(h2o_molecule, size=30., n_structures=200)
    return setup

@pytest.fixture()
def exported_file_path():
    """Returns the file path of the exported input file"""
    return os.path.join(_ABS_DIR_PATH, "exported_input_file.inp")

@pytest.fixture()
def export_input_file(simple_packmol_input_system, h2o_molecule, exported_file_path):
    """Exports the input file based on a simple packmol system"""
    exporter = PackmolInputExporter(exported_file_path)
    with exporter:
        exporter.write(setup=simple_packmol_input_system,
                       structure_file_names={h2o_molecule:"water.pdb"},
                       output_name="output_example.pdb")

def test_correct_name_for_each_structure(export_input_file, exported_file_path):
    """Tests that the correct name has been applied for each structure"""
    with open(exported_file_path, "r") as input_file:
        for line in input_file:
            if line.startswith("structure"):
                assert line.endswith("water.pdb\n")

def test_correct_system_parameters_exported(export_input_file, exported_file_path):
    """Tests that the right system parameters are exported in the right places"""
    with open(exported_file_path, "r") as input_file:
        for line in input_file:
            if line.startswith("inside cube"):
                assert line.endswith("0.0 0.0 0.0 30.0")
            elif line.startswith("number"):
                assert line.endswith("200")


def test_correct_structures_format(export_input_file, exported_file_path):
    """
    Test that the structures begin and end with the right keywords,
    and that the right indent is applied.
    """

    full_file_text = ""
    with open(exported_file_path, "r") as input_file:
        for line in input_file:
            full_file_text += line

    structure_pattern = re.compile("structrue")
    end_structure_pattern = re.compile("end structrue")

    if structure_pattern.search(full_file_text):
        assert end_structure_pattern.search(full_file_text)
