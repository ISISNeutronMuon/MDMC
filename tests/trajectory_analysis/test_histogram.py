"""Tests for configuration, trajectory and histogram calculation"""

import pytest
import numpy as np
from copy import deepcopy

import MDMC.trajectory_analysis.trajectory as trj

from tests.MD.test_simulation import universe, atom, water_molecule, \
    water_SPCE_universe, UNIVERSE_DIMENSIONS

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
    conf_sum_list = conf_sum.structure_list
    for structure in conf_copy.structure_list + configuration.structure_list:
        assert structure in conf_sum_list
        conf_sum_list.remove(structure)
    assert len(conf_sum_list) == 0

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
