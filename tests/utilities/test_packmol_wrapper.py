"""Tests for the packmol wrapper utility of MDMC"""
import pytest
from pytest_cases import fixture

import MDMC.utilities.packmol_wrapper as packmol
import tests.test_data.data as test_data
from MDMC.MD import Universe

# lammps mark used to ensure test runs in docker container
pytestmark = [pytest.mark.lammps]

@fixture
def packmol_data_path():
    """
    Returns the path to the directory for the packmol configuration files
    (in the docker container)
    """
    return test_data._ABS_DIR_PATH + test_data._PACKMOL_PATH

@fixture
def filled_universe(packmol_data_path):
    """Returns the universe filled by packmol"""
    return packmol.fill_with_packmol("bilayer.inp", packmol_data_path)

@fixture
def input_file_name():
    return "bilayer.inp"

def test_get_packmol_path():
    """Tests that the packmol path is correct within the docker container"""
    actual_path = packmol.get_packmol_path()
    correct_path = "/opt/other/packmol/packmol-20.14.0/packmol"
    assert actual_path == correct_path

def test_packmol_result_is_identical_between_runs(filled_universe, packmol_data_path, input_file_name):
    universe_1 = filled_universe
    universe_2 = packmol.fill_with_packmol(input_file_name, packmol_data_path)
    assert dir(universe_1) == dir(universe_2)
def test_returns_universe(filled_universe):
    assert type(filled_universe) is Universe

def test_correct_system_properties(filled_universe):
    assert filled_universe.n_atoms == 8000
    assert filled_universe.n_molecules == 1100
    assert set(filled_universe.element_list) == {"H", "O", "C"}
    # Double bonds are not represented in MDMC, so there should be 7000 bonds, but 6900 are seen
    assert len(filled_universe.bonded_interactions) == 6900

def test_incorrect_packmol_path(packmol_data_path, input_file_name):
    """Tests that a packmol run with an incorrect path will return an error"""
    incorrect_path = packmol_data_path + "/incorrect_place"
    with pytest.raises(IOError):
        packmol.fill_with_packmol(input_file_name, incorrect_path)

def test_incorrect_packmol_filename(packmol_data_path, input_file_name):
    """Tests that a packmol run with an incorrect filename will return an error"""
    incorrect_path = packmol_data_path + "/incorrect_place"
    with pytest.raises(IOError):
        packmol.fill_with_packmol(input_file_name, incorrect_path)

def test_get_packmol_output_name(packmol_data_path, input_file_name):
    """Tests that the output name of a packmol file will be correctly retrieved"""
    actual_name = packmol.get_packmol_output_name(packmol_data_path+"/"+input_file_name)
    correct_name = "bilayer.pdb"
    assert actual_name == correct_name

def test_get_packmol_universe_dimensions(packmol_data_path, input_file_name):
    """Tests that the dimensions are correctly read from the packmol input"""
    actual_dim = packmol.get_packmol_universe_dimensions(packmol_data_path+"/"+input_file_name)
    correct_dim = [72., 60., 60.,]
    assert actual_dim == correct_dim
def test_call_external_program():
    """Tests that the external programs are called correctly"""
    pass
