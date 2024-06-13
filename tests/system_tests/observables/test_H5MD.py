from pathlib import Path

try:
    import cPickle as pickle
except ImportError:
    import pickle
import zlib

import h5py
import numpy as np

from MDMC.writers import H5MD_build
from MDMC.common import units, atom_properties
from tests.system_tests.observables.data_manager import trajectory
from tests.test_data import data
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory
from MDMC.readers import H5MD_reader

FILE_NAME = 'test_file'
FILE_PATH = Path(FILE_NAME).with_suffix('.h5')
ROOT_LOC = 'particles/simulation'

def setup_module(_):

    """Create H5MD file from compact trajectory
    """

    # Unzip and unpickle the trajectory
    with open(data.OBJECT_DATA['compact_trajectory'], 'rb') as compressed_trajectory:
        pickled_trajectory = zlib.decompress(compressed_trajectory.read())
    H5MD_build.build_full(pickle.loads(pickled_trajectory, encoding='latin-1'), 
                          FILE_NAME, 
                          timestamp=False)

def teardown_module(_):

    """Deletes H5MD file after testing
    """

    FILE_PATH.unlink(missing_ok=False)

def test_read_masses(trajectory):
    """Tests that the particle masses are stored the H5MD file are the same as in the trajectory
    """
    expected_masses = trajectory.atom_masses
    with h5py.File(FILE_PATH, 'r') as file:
        h5md_mass = H5MD_reader.read_atom_mass(file)
    assert np.array_equal(h5md_mass, expected_masses)
        
def test_read_mass_unit():
    """Tests that the particle mass unit stored the H5MD file is the same as in the trajectory
    """
    expected_unit = str(units.SYSTEM['MASS'])
    with h5py.File(FILE_PATH, 'r') as file:
        h5md_unit = H5MD_reader.read_atom_mass_unit(file)
    assert h5md_unit == expected_unit

def test_read_time(trajectory):
    """Tests That the correct time can be calculated from what is stored in the H5MD file
    """
    expected_times = trajectory.times
    with h5py.File(FILE_PATH, 'r') as file:
        h5md_time = []
        h5md_time = [H5MD_reader.read_times(file, step)
                     for step in range(trajectory.n_steps)]
    assert np.array_equal(h5md_time, expected_times)

def test_time_units(trajectory):
    """Tests that the time unit stored the H5MD file is the same as in the trajectory
    """
    expected_unit = trajectory.time_unit
    with h5py.File(FILE_PATH, 'r') as file:
        h5md_unit = H5MD_reader.read_time_unit(file)
    assert h5md_unit == expected_unit

def test_read_n_steps(trajectory):
    """Tests That the correct number of steps are stored the H5MD file are the same as in the trajectory
    """
    expected_n_steps = trajectory.n_steps
    with h5py.File(FILE_PATH, 'r') as file:
        h5md_n_steps = H5MD_reader.read_number_steps(file)
    assert h5md_n_steps == expected_n_steps

def test_read_dimensions(trajectory):
    """Tests that the box dimensions stored in the H5MD file are the same as the trajectory
    """
    expected_dimensions = len(trajectory.dimensions)
    with h5py.File(FILE_PATH, 'r') as file:
        h5md_dimensions = H5MD_reader.read_box_dimension(file)
    assert np.array_equal(expected_dimensions, h5md_dimensions)

def test_read_species(trajectory):
    """Tests that the particle masses are consistent across particles of the same species
    """
    expected_species = []
    for element in trajectory.element_list:
        expected_species.append(list(atom_properties.ATOMIC_NUMBER.keys())[list(atom_properties.ATOMIC_NUMBER.values()).index(element)])
    with h5py.File(FILE_PATH, 'r') as file:
        h5md_species = H5MD_reader.read_species(file)
    assert np.array_equal(expected_species, h5md_species)

def test_read_positions(trajectory):
    """Tests that the positions of the particles stored in the H5MD file are the same as the trajectory
    """
    expected_positions = trajectory.position
    with h5py.File(FILE_PATH, 'r') as file:
        h5md_positions = H5MD_reader.read_positions(file)
    assert np.array_equal(expected_positions, h5md_positions)

def test_positions_units(trajectory):
    """Tests that the particle positions unit stored the H5MD file is the same as in the trajectory
    """
    expected_unit = trajectory.position_unit
    with h5py.File(FILE_PATH, 'r') as file:
        h5md_unit = H5MD_reader.read_positions_unit(file)
    assert h5md_unit == expected_unit

def test_read_velocity(trajectory):
    """Tests that the particle velosity stored in H5MD file is the same as the trajectorys velosity
    """
    expected_velocity = trajectory.velocity
    with h5py.File(FILE_PATH, 'r') as file:
        h5md_velocity = H5MD_reader.read_velocity(file)
    assert np.array_equal(expected_velocity, h5md_velocity)

def test_velocity_units(trajectory):
    """Tests that the particle velocity unit stored the H5MD file is the same as in the trajectory
    """
    expected_unit = trajectory.velocity_unit
    with h5py.File(FILE_PATH, 'r') as file:
        h5md_unit = H5MD_reader.read_velocity_unit(file)
    assert h5md_unit == expected_unit

def test_read_charge(trajectory):
    """Tests that the particle charges are stored in the H5MD file are the same as in the trajectory
    """
    expected_charge = trajectory.atom_charges
    expected_charge[expected_charge == None] = 0
    with h5py.File(FILE_PATH, 'r') as file:
        h5md_charge = H5MD_reader.read_charge(file)
    assert np.array_equal(expected_charge, h5md_charge)

def test_charge_units():
    """Tests that the particle charge unit stored the H5MD file is the same as in the trajectory
    """
    expected_unit = str(units.SYSTEM['CHARGE'])
    with h5py.File(FILE_PATH, 'r') as file:
        h5md_unit = H5MD_reader.read_charge_unit(file)
    assert h5md_unit == expected_unit

def test_atom_symbols(trajectory):
    expected_symbols = trajectory.element_list
    with h5py.File(FILE_PATH, 'r') as file:
        h5md_charge = H5MD_reader.read_atom_symbols(file)
    assert np.array_equal(expected_symbols, h5md_charge)

def test_tragect_from_file():
    """Tests that the H5MD file can be read back in as a compact tragectory
    """
    CompactTrajectory.create_from_file(FILE_PATH)
    