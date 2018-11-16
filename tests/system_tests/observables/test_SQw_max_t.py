"""System tests for total, coherent and incoherent SQw calculations with
maximum times shorter than the provided trajectories

The maximum time, t, for a required trajectory for calculating SQw depends on
the SQw energy step size, dE.  If the trajectory provided has a larger t than is
required by dE, SQw must still be calculated for dE step sizes.  These unit
tests ensure that SQw is the same (within uncertainty) independent of the
trajectory length, it the same energies are specified.

AUTHOR :    Thomas Farmer        START DATE :    15/11/2018, 17:11:21"""

try:
    import cPickle as pickle
except:
    import pickle
import zlib

import numpy as np
from numpy.testing import assert_allclose
import pytest

from MDMC.trajectory_analysis.observables.obs_factory import ObservableFactory
from tests.test_data import data


@pytest.fixture(scope="module")
def trajectory():

    """
    Read the trajectory

    Trajectory is unzipped and unpickled.
    """

    compressed_trajectory = open(data.OBJECT_DATA['trajectory'], 'r').read()
    pickled_trajectory = zlib.decompress(compressed_trajectory)
    trajectory = pickle.loads(pickled_trajectory)
    return trajectory


@pytest.fixture(scope="module")
def independent_variables():

    """
    Returns the independent variables required for SQw, SQw_coh, and SQw_incoh
    """

    Q = np.arange(1.6, 21, 1.6)
    E = np.arange()

    """
    Read trajectory into REPL.
    Create SQw with half of the trajectory.
    Determine E values - use these above.
    """
