"""Tests for the H5MD file are writing and reading the files correctly.

Notes
-----
The few tests that are not part of the @pytest.mark.parametrize 
are because they are read in fromthe H5MD or the compact trajectory 
diferenty from the other variables
"""
from pathlib import Path

try:
    import cPickle as pickle
except ImportError:
    import pickle
import zlib

import pytest
import h5py
import numpy as np
import periodictable

from MDMC.writers import H5MD_build
from MDMC.common import units
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

@pytest.mark.parametrize("expected, test_input", [("atom_masses", "mass"), 
                                                 ("position", "position"), 
                                                 ("velocity", "velocity"), 
                                                 ("element_list", "atom_symbols"),
                                                 ("atom_charges", "charge")])
def test_H5MD_array(trajectory, expected, test_input):
    """Tests that everything stored in the H5MD file that is in an array is stored as expected
    """
    expected = getattr(trajectory, expected)

    if test_input == "charge":
        expected[expected == None] = 0 # Covers if for testing charge that is sometimes None's

    with h5py.File(FILE_PATH, 'r') as file:
        h5md_read = H5MD_reader.read_dataset(file, test_input)

    assert np.array_equal(h5md_read, expected)

@pytest.mark.parametrize("expected, test_input", [("MASS", "mass"), 
                                                 ("TIME", "time"), 
                                                 ("LENGTH", "position"), 
                                                 ("CHARGE", "charge")])
def test_H5MD_units(expected, test_input):
    """Tests that all units in the H5MD file are stored as expected
    """
    expected = str(units.SYSTEM[expected])
    with h5py.File(FILE_PATH, 'r') as file:
        h5md_read = H5MD_reader.read_units(file, test_input)
    assert expected == h5md_read

def test_velocity_units():
    """Tests that velocity units stored in the H5MD file is as expected
    """
    expected = units.SYSTEM['LENGTH'] / units.SYSTEM['TIME']
    with h5py.File(FILE_PATH, 'r') as file:
        h5md_read = H5MD_reader.read_units(file, "velocity")
    assert expected == h5md_read

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

def test_read_time(trajectory):
    """Tests That the correct time can be calculated from what is stored in the H5MD file
    """
    expected_times = trajectory.times
    with h5py.File(FILE_PATH, 'r') as file:
        h5md_time = []
        h5md_time = [H5MD_reader.read_times(file, step)
                     for step in range(trajectory.n_steps)]
    assert np.array_equal(h5md_time, expected_times)

def test_tragect_from_file():
    """Tests that the H5MD file can be read back in as a compact tragectory
    """
    CompactTrajectory.create_from_file(FILE_PATH)
