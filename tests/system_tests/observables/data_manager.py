"""Manages test data for system tests
"""

try:
    import cPickle as pickle
except ImportError:
    import pickle
import zlib

import pytest

from tests.test_data import data
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory

pytestmark = [pytest.mark.lammps]

@pytest.fixture(scope="session")
def trajectory():

    """
    Returns
    -------
    CompactTrajectory
        A 50 configuration trajectory (from a 50000 step simulation) of 2048
        water molecules
    """

    # Unzip and unpickle the trajectory
    compressed_trajectory = open(data.OBJECT_DATA['compact_trajectory'], 'rb').read()
    pickled_trajectory = zlib.decompress(compressed_trajectory)
    return pickle.loads(pickled_trajectory, encoding='latin-1')


@pytest.fixture(scope="module")
def Q_vectors():

    """
    Returns
    -------
    array
        An array of arrays of Q vectors for each Q value, which is used for all
        Q vector calculations in nMOLDYN.
    """

    return pickle.load(open(data.OBS_DATA['Q_vectors'], 'rb'),
                       encoding='latin-1')
