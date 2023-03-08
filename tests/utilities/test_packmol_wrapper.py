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

# def test_correct_packmol_run(packmol_data_path):
#     """
#     Tests that a correctly-initiated packmol run will
#     correctly run packmol and read in the result
#     """
#
#     pass

def test_packmol_result_is_identical_between_runs(filled_universe, packmol_data_path, input_file_name):
    universe_1 = filled_universe
    universe_2 = packmol.fill_with_packmol(input_file_name, packmol_data_path)
    assert dir(universe_1) == dir(universe_2)
def test_returns_universe(filled_universe):
    assert type(filled_universe) == Universe

def test_correct_atoms(filled_universe):
    assert filled_universe.n_atoms == 7187
    assert filled_universe.n_molecules == 1199
    assert set(filled_universe.element_list) == {"H", "O", "C"}

def test_molecules_are_connected(filled_universe):
    assert len(filled_universe.bonded_interactions) == 7150

def test_incorrect_packmol_path(packmol_data_path, input_file_name):
    """Tests that a packmol run with an incorrect path will return an error"""
    incorrect_path = packmol_data_path + "/incorrect_place"
    with pytest.raises(IOError) as except_info:
        packmol.fill_with_packmol(input_file_name, incorrect_path)

def test_incorrect_packmol_filename(packmol_data_path, input_file_name):
    """Tests that a packmol run with an incorrect filename will return an error"""
    incorrect_path = packmol_data_path + "/incorrect_place"
    with pytest.raises(IOError) as except_info:
        packmol.fill_with_packmol(input_file_name, incorrect_path)


def test_get_packmol_output_name():
    """Tests that the output name of a packmol file will be correctly retrieved"""
    pass

def test_get_packmol_universe_dimensions():
    """Tests that the dimensions are correctly read from the packmol input"""
    pass

def test_call_external_program():
    """Tests that the """
    pass

def test_unrelated():
    from MDMC.readers.configurations import pdb
    from MDMC.MD.structures import Molecule
    reader = pdb.ProteinDataBankReader(test_data._ABS_DIR_PATH + test_data._PACKMOL_PATH + "/water.pdb")
    reader.__enter__()
    reader.parse()
    water = Molecule(atoms=reader.atoms, interactions=reader.bonds)
    assert water.bonded_interactions == 2