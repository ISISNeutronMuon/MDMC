"""Tests for configuration, trajectory and histogram calculation

AUTHOR :    Thomas Farmer        START DATE :    2018-5-29 17:36:22"""

import pytest
import numpy as np

import MDMC.src.trajectory_analysis.trajectory as trj

from MDMC.tests.MD.test_simulation import universe, atom, water_molecule, \
    water_SPCE_universe, UNIVERSE_DIMS

R_AXIS = [0., 20., 0.5]
T_AXIS = [0., 5., 1.0]

TRAJ_TIME_START = 0.
TRAJ_TIME_END = 5.0
TRAJ_TIME_STEP = 0.5
TIMES = np.arange(TRAJ_TIME_START, TRAJ_TIME_END, TRAJ_TIME_STEP)

@pytest.fixture
def configuration(water_SPCE_universe):
    return trj.TemporalConfiguration(0., *water_SPCE_universe.atom_list)

@pytest.fixture
def trajectory(water_SPCE_universe):

    """
    A list of identical configurations with different times is produced. This
    is passed to Trajectory.
    """

    configurations = []
    for time in TIMES:
        configurations.append(trj.TemporalConfiguration(
            time, *water_SPCE_universe.configuration.atom_list))
    return trj.Trajectory(*configurations)

@pytest.fixture
def histogram(trajectory):
    return trj.Histogram(trajectory, r = R_AXIS, time = T_AXIS)

def test_configuration(configuration):
    pass

def test_trajectory(trajectory):

    """
    Test for:

    Existance of times, atom list, positions, velocities
    Ensure structuredarray maintains order - test filter configs by time against more explicit version (i.e. both taken from data)
    """

    pass

def test_histogram(configuration, histogram):

    """
    Test for:

    number of elements of histogram - for the first time bin this should be
    equal to the sum of the pairwise distances (i.e. range(len(configuration)))
    multiplied by the number of histograms grouped in the bin (i.e. the time
    step of the trajectories divided by the histogram time bin size)
    equal bin sizes
    number of frames
    """

    assert sum(range(len(configuration))) == \
        int(sum(histogram.data['histogram'][0][0]) / (
        T_AXIS[2] / TRAJ_TIME_STEP))
