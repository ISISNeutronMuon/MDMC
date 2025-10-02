"""Tests for the H5MD file are writing and reading the files correctly.

Notes
-----
The few tests that are not part of the @pytest.mark.parametrize
are because they are read from the H5MD or CompactTrajectory
differently to the other variables.
"""
from pathlib import Path

try:
    import cPickle as pickle
except ImportError:
    import pickle

import zlib

import h5py
import numpy as np
import pytest

from tests.system_tests.observables.data_manager import trajectory
from tests.test_data import data

from MDMC.common import units
from MDMC.exporters.trajectories import H5MD_build
from MDMC.readers import H5MD_reader
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory

FILE_NAME = Path('test_file.h5')
ROOT_LOC = 'particles/simulation'

@pytest.fixture(scope="module")
def traj():
    # Unzip and unpickle the trajectory
    with open(data.OBJECT_DATA['compact_trajectory'], 'rb') as compressed_trajectory:
        pickled_trajectory = zlib.decompress(compressed_trajectory.read())

    return pickle.loads(pickled_trajectory, encoding='latin-1')

@pytest.fixture(scope="module")
def open_file(traj, tmp_path_factory):
    """
    Create H5MD file from CompactTrajectory and return it.

    Yields
    ------
    h5py.File
        An open H5MD file for testing.
    """

    file_path = Path(tmp_path_factory.getbasetemp())

    H5MD_build.write_H5MD(traj,
                          filename=FILE_NAME,
                          timestamp=False,
                          file_loc=file_path,
                          creator_name="TEST",
                          creator_email="EMAIL")

    file = h5py.File(file_path / FILE_NAME, 'r')
    yield file
    file.close()

@pytest.mark.parametrize("expected, test_input", [("atom_masses", "mass"),
                                                  ("position", "position"),
                                                  ("velocity", "velocity"),
                                                  ("element_list", "atom_symbols"),
                                                  ("atom_charges", "charge")])
def test_H5MD_array(open_file, trajectory, expected, test_input):
    """Tests that everything stored in the H5MD file that is in an array is stored as expected."""
    expected = getattr(trajectory, expected)

    if test_input == "charge":
        for count, value in enumerate(expected):
            if value is None:
                expected[count] = 0.0 # Covers if for testing charge that is sometimes None's

    h5md_read = H5MD_reader.read_dataset(open_file, test_input)

    assert np.array_equal(h5md_read, expected)

@pytest.mark.parametrize("creator",
                         [{},
                          {"creator_name": "TEST"},
                          {"creator_email": "TEST"}]
                         )
def test_H5MD_requires_creator(tmp_path, traj, creator):
    """Test create fails if user details not provided."""
    with pytest.raises(ValueError, match="No creator_name"):
        H5MD_build.write_H5MD(traj,
                              filename=FILE_NAME,
                              timestamp=False,
                              file_loc=tmp_path,
                              **creator)

@pytest.mark.parametrize("expected, test_input",
                         [(units.SYSTEM["MASS"], "mass"),
                          (units.SYSTEM["TIME"], "time"),
                          (units.SYSTEM["LENGTH"], "position"),
                          (units.SYSTEM["CHARGE"], "charge"),
                          (units.SYSTEM["LENGTH"] / units.SYSTEM["TIME"], "velocity")
                          ])
def test_H5MD_units(open_file, expected, test_input):
    """Tests that all units in the H5MD file are stored as expected."""
    h5md_read = H5MD_reader.read_units(open_file, test_input)
    assert str(expected) == h5md_read

def test_read_n_steps(open_file, trajectory):
    """Tests the number of steps stored in the H5MD file are same as trajectory."""
    expected_n_steps = trajectory.n_steps
    h5md_n_steps = H5MD_reader.read_number_steps(open_file)
    assert h5md_n_steps == expected_n_steps

def test_read_dimensions(open_file, trajectory):
    """Tests that the box dimensions stored in the H5MD file are the same as the trajectory."""
    expected_dimensions = len(trajectory.dimensions)
    h5md_dimensions = H5MD_reader.read_box_property(open_file, "dimensions")
    assert np.array_equal(expected_dimensions, h5md_dimensions)

def test_read_time(open_file, trajectory):
    """Tests That the correct time can be calculated from what is stored in the H5MD file."""
    expected_times = trajectory.times
    h5md_time = [H5MD_reader.read_times(open_file, step)
                 for step in range(trajectory.n_steps)]
    assert np.array_equal(h5md_time, expected_times)

def test_trajectory_from_file(tmp_path_factory, trajectory):
    """Tests that the H5MD file can be read back in as a compact trajectory."""
    file_path = Path(tmp_path_factory.getbasetemp())
    file_path = file_path / FILE_NAME

    new_ct = CompactTrajectory.create_from_h5md(file_path)
    assert np.array_equal(trajectory.position, new_ct.position)
    assert np.array_equal(trajectory.time, new_ct.time)
    assert np.array_equal(trajectory.position_unit, new_ct.position_unit)
