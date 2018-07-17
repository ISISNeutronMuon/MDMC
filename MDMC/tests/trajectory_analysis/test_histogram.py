"""Tests for configuration, trajectory and histogram calculation

AUTHOR :    Thomas Farmer        START DATE :    2018-5-29 17:36:22"""

import pytest
import numpy as np
from copy import deepcopy

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

    # TODO: Change the configurations to scaled copies once configuration scaling has been implemented

    configurations = []
    for time in TIMES:
        configurations.append(trj.TemporalConfiguration(
            time, *water_SPCE_universe.configuration.atom_list))
    return trj.Trajectory(*configurations)

@pytest.fixture
def histogram(trajectory):
    return trj.Histogram(trajectory, r = R_AXIS, time = T_AXIS)

def test_configuration(configuration):

    """
    Test for:

    Existence of atom_list, atom_positions, atom_velocities, structure_list
    Add configurations
    Filter
    time
    """

    # Copy the configuration, sum the original and the copy, test that the
    # structure_list in the sum is exactly composed of the structures in the
    # original and the copy.
    conf_copy = deepcopy(configuration)
    conf_sum = configuration + conf_copy
    for structure in conf_copy.structures_list + configuration.structures_list:
        assert structure in conf_sum.structures_list
        conf_sum.structures_list.remove(structure)
    assert len(conf_sum.structures_list) == 0

    # Testing filter_by_element
    H_atoms = configuration.filter_by_element('H')
    for atom in H_atoms:
        assert atom.element == 'H'
    for atom in set(configuration.atom_list) - set(H_atoms):
        assert atom.element != 'H'

def test_trajectory(trajectory):

    """
    Test for:

    Existence of times, atom_list, positions, velocities
    filter_by_time results in expected time for the new trajectory
    """

    START = 1
    STOP = 4
    single_time_traj = trajectory.filter_by_time(trajectory.times[START])
    slice_time_traj = trajectory.filter_by_time(trajectory.times[START],
        trajectory.times[STOP])

    assert single_time_traj.times == trajectory.times[START]
    assert np.all(slice_time_traj.times) == np.all(trajectory.times[START:STOP])


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
