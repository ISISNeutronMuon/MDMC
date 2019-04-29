"""System tests for total, coherent and incoherent SQw calculations with
maximum times shorter than the provided trajectories

The maximum time, t, for a required trajectory for calculating SQw depends on
the SQw energy step size, dE.  If the trajectory provided has a larger t than is
required by dE, SQw must still be calculated for dE step sizes.  These unit
tests ensure that SQw is the same (within uncertainty) independent of the
trajectory length, it the same energies are specified.  THIS MODULE COULD BE
PARAMETERIZED TO TEST OTHER OBSERVABLES

AUTHOR :    Thomas Farmer        START DATE :    15/11/2018, 17:11:21"""

try:
    import cPickle as pickle
except:
    import pickle
import zlib

import numpy as np
from numpy.testing import assert_allclose
import pytest

from MDMC.common.constants import h_bar
from MDMC.trajectory_analysis.observables.obs_factory import ObservableFactory
from tests.test_data import data


@pytest.fixture(scope="module")
def trajectory():

    """
    Trajectory is read, unzipped and unpickled.

    Returns:
    Trajectory
    """

    compressed_trajectory = open(data.OBJECT_DATA['trajectory'], 'r').read()
    pickled_trajectory = zlib.decompress(compressed_trajectory)
    trajectory = pickle.loads(pickled_trajectory)
    return trajectory


@pytest.fixture(scope="module")
def independent_variables(trajectory):

    """
    Calculate the independent variables

    E is equivalent to the times from half the trajectory length

    Returns:
    Dictionary of independent variables required for SQw, SQw_coh, and SQw_incoh
    """

    # Use half the trajectory steps to calculate the Energies
    n = len(trajectory.times) / 2
    dt = trajectory.times[1] - trajectory.times[0]
    E = h_bar * 1e18 * np.pi * np.arange(n) / (n * dt)
    Q = np.arange(1.6, 21, 1.6)

    return {'E':E, 'Q':Q}


@pytest.fixture(params=['SQw', 'SQw_coh', 'SQw_incoh'])
def SQw_type(request):

    """
    SQw_type is parameterized with the strings required to create SQw, SQw_coh
    and SQw_incoh observable types
    """

    return request.param


def test_SQw_max_t(trajectory, independent_variables, SQw_type):

    """
    Tests the total SQw with times shorter than provided the trajectory

    Three SQw are calculated, one using the full trajectory, one using the first
    half of the trajectory, and one using the second half of the trajectory.
    All SQw are calculated for the same values of Q and E.  The SQw calculated
    from the total trajectory is tested for consistency with the two half
    trajectory SQws.
    """

    T_RES = 30.
    DIMS = [39.42210674, 39.42210674, 39.42210674]

    SQw_full = ObservableFactory.create_observable(SQw_type)
    SQw_1 = ObservableFactory.create_observable(SQw_type)
    SQw_2 = ObservableFactory.create_observable(SQw_type)

    for SQw in [SQw_full, SQw_1, SQw_2]:
        SQw.independent_variables = independent_variables

    n = len(trajectory.times) / 2
    SQw_full.calculate_from_MD(trajectory, t_resolution=T_RES, dims=DIMS)
    SQw_1.calculate_from_MD(trajectory[:n], t_resolution=T_RES, dims=DIMS)
    SQw_2.calculate_from_MD(trajectory[n:], t_resolution=T_RES, dims=DIMS)

    # Calculate the total standard deviation for the two half runs and test that
    # the total run is within a factor of 3
    SQw_1_2_mean = np.mean([SQw_1.SQw, SQw_2.SQw], axis=0)
    stdev = np.std([SQw_1.SQw, SQw_2.SQw], axis=0)
    stdev_total = np.sum(stdev)
    stdev_full = np.std([SQw_1_2_mean, SQw_full.SQw], axis=0)
    assert np.sum(stdev_full) < 3 * stdev_total

    # Test that the stdev for each Q,w value for the total run is within a
    # factor of 2 of the maximum standard deviation of any point
    assert np.all(stdev_full < 2 * np.max(stdev))
