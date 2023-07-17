"""Tests for the packmol wrapper utility of MDMC"""
import os.path
import shutil

import numpy as np
import pytest

import MDMC.MD.packmol.packmol_wrapper as packmol
from MDMC.MD import Universe, Atom, BondAngle, Molecule, Bond
from MDMC.MD.packmol.packmol_setup import PackmolSetup

# lammps mark used to ensure test runs in docker container
pytestmark = [pytest.mark.lammps]

@pytest.fixture()
def h2o_molecule():
    """A water molecule."""
    h_1 = Atom("H")
    h_2 = Atom('H', position=[0., 1.63298, 0.])
    o_1 = Atom('O', position=[0., 0.81649, 0.57736])
    ho_bonds = Bond((h_1, o_1), (h_2, o_1))
    h20_bondangle = BondAngle(atom_tuples=[(h_1, o_1, h_2)])
    h2o_mol = Molecule(atoms=[h_1, h_2, o_1], interactions=[h20_bondangle, ho_bonds])
    return h2o_mol

@pytest.fixture()
def water_setup(h2o_molecule):
    setup = PackmolSetup()
    setup.add_cube(h2o_molecule, size=30., density=0.05)
    return setup

@pytest.fixture()
def water_argon_mix_setup(h2o_molecule, water_setup):
    water_setup.add_cube(Atom("Ar"), size=40., origin=(30., 30., 30.), density=0.05)
    return water_setup

@pytest.fixture()
def simple_filled_universe(water_setup):
    """Returns the universe result from a packmol run using the water_setup setup object"""
    return packmol.PackmolFiller(water_setup).fill_with_packmol()

@pytest.fixture()
def complex_filled_universe(water_setup):
    """Returns the universe result from a packmol run using the water_argon_mix_setup setup object"""
    return packmol.PackmolFiller(water_argon_mix_setup).fill_with_packmol()

@pytest.fixture()
def simple_filled_universe_filler_object(water_setup):
    """A `PackmolFiller` Object after running a fill run"""
    filler = packmol.PackmolFiller(water_setup)
    filler.fill_with_packmol()
    return filler


def test_packmol_result_is_identical_between_runs(simple_filled_universe, water_setup):
    """Test that filling is deterministic when using the same setup"""
    universe_1 = simple_filled_universe
    universe_2 = packmol.PackmolFiller(water_setup).fill_with_packmol()
    assert dir(universe_1) == dir(universe_2)

def test_returns_universe(simple_filled_universe):
    """ Tests that the result of filling with packmol will be a `Universe` object"""
    assert type(simple_filled_universe) is Universe

def test_get_packmol_output_name(simple_filled_universe_filler_object):
    """Tests that the output name of a packmol file will be correctly retrieved"""
    actual_path = simple_filled_universe_filler_object.get_packmol_output_path()
    actual_name = os.path.basename(actual_path)
    correct_name = "output-universe.pdb"
    assert actual_name == correct_name

def test_get_packmol_universe_dimensions(water_argon_mix_setup):
    """Tests that the dimensions are correctly read from the packmol input"""
    filled_universe = packmol.PackmolFiller(water_argon_mix_setup).fill_with_packmol()
    actual_dim = filled_universe.dimensions
    correct_dim = [70.,70.,70.]
    assert np.allclose(actual_dim, correct_dim)

def test_directories_created_correctly(simple_filled_universe_filler_object):
    """Tests that the needed directory/ies are created for the packmol files"""
    packmol_files_path = simple_filled_universe_filler_object.get_packmol_files_path()
    assert os.path.exists(packmol_files_path)
    assert os.path.basename(packmol_files_path) == "packmol_files"

def test_all_files_are_created_after_run(simple_filled_universe_filler_object):
    """Tests that the right files are created for the packmol run as necessary"""
    packmol_files_path = simple_filled_universe_filler_object.get_packmol_files_path()
    input_file_path = os.path.join(packmol_files_path, "input_file.inp")
    output_file_path = os.path.join(packmol_files_path, "output-universe.pdb")
    mol_name = [molecule.name for molecule in simple_filled_universe_filler_object.setup_data.get_structures()]
    assert os.path.exists(input_file_path)
    assert os.path.exists(output_file_path)
    assert os.path.isfile(input_file_path)
    assert os.path.isfile(output_file_path)
    for i, mol_name in enumerate(mol_name):
        mol_path = os.path.join(packmol_files_path, f"{str(mol_name)}-{str(i)}.pdb")
        assert os.path.exists(mol_path)
        assert os.path.isfile(mol_path)

def test_packmol_program_path(simple_filled_universe_filler_object):
    """Tests to make sure that the file to the packmol program is as reported by the shell"""
    packmol_path = simple_filled_universe_filler_object.get_packmol_path()
    assert shutil.which("packmol") == packmol_path

def test_correct_system_properties(simple_filled_universe):
    """Tests to make sure the correct system properties are in the final filled system"""
    assert simple_filled_universe.n_atoms == 4050
    assert simple_filled_universe.n_molecules == 1350
    assert set(simple_filled_universe.element_list) == {"H", "O"}
    # 2 bonded interactions per molecule
    assert len(simple_filled_universe.bonded_interactions) == 2700

def test_complex_system_properties(complex_filled_universe):
    """Tests to make sure the correct system properties are in the final complex system"""
    assert simple_filled_universe.n_atoms == 5900
    assert simple_filled_universe.n_molecules == 1350
    assert set(simple_filled_universe.element_list) == {"H", "O", "Ar"}
    # 2 bonded interactions per molecule
    assert len(simple_filled_universe.bonded_interactions) == 4050