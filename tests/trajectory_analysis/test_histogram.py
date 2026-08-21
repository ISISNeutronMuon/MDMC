"""Tests for configuration, trajectory and histogram calculation"""

import pytest
import numpy as np
from copy import deepcopy

import MDMC.trajectory_analysis.trajectory as trj
import MDMC.trajectory_analysis.compact_trajectory as ctrj

from tests.MD.test_simulation import universe, atom, water_molecule, \
    water_OPLSAA_universe, UNIVERSE_DIMENSIONS

R_AXIS = [0., 20., 0.5]
T_AXIS = [0., 5., 1.0]

TRAJ_TIME_START = 0.
TRAJ_TIME_END = 5.0
TRAJ_TIME_STEP = 0.5
TIMES = np.arange(TRAJ_TIME_START, TRAJ_TIME_END, TRAJ_TIME_STEP)

@pytest.fixture
def configuration(water_OPLSAA_universe):
    return trj.TemporalConfiguration(0., *water_OPLSAA_universe.atoms)

@pytest.fixture
def trajectory(water_OPLSAA_universe):

    """
    A list of identical configurations with different times is produced. This
    is passed to Trajectory.
    """
    
    n_atoms = len(water_OPLSAA_universe.configuration.atoms)
    n_steps = len(TIMES)
    temp_traj = ctrj.configurations_as_compact_trajectory(*[water_OPLSAA_universe.configuration])
    traj = ctrj.CompactTrajectory(n_steps, n_atoms)
    for step_num, time in enumerate(TIMES):
        traj.writeOneStep(step_num= step_num,
                          time= time,
                          positions= temp_traj.position[0])
    return traj

def test_configuration(configuration):

    """
    Test for:

    Existence of atoms, atom_positions, atom_velocities, structure_list
    Add configurations
    Filter
    time
    """

    # Copy the configuration, sum the original and the copy, test that the
    # structure_list in the sum is exactly composed of the structures in the
    # original and the copy.
    conf_copy = deepcopy(configuration)
    conf_sum = configuration + conf_copy
    # Introducing the getter for structure_list (and in doing so, creating
    # _structure_list) for Configuration objects means that we need to remove
    # elements from a local variable (conf_sum_list) rather than directly from
    # conf_sum.structure_list, as removing from the latter will have no effect
    # the underlying _structure_list of weak references.
    conf_sum_list = conf_sum.structure_list
    for structure in conf_copy.structure_list + configuration.structure_list:
        assert structure in conf_sum_list
        conf_sum_list.remove(structure)
    assert len(conf_sum_list) == 0

    # Testing filter_by_element
    H_atoms = configuration.filter_by_element('H')
    for atom in H_atoms:
        assert atom.element.symbol == 'H'
    for atom in set(configuration.atoms) - set(H_atoms):
        assert atom.element.symbol != 'H'

def test_trajectory(trajectory):

    """
    Test for:

    Existence of times, atoms, positions, velocities
    filter_by_time results in expected time for the new trajectory
    """

    START = 1
    STOP = 4
    single_time_traj = trajectory.filter_by_time(trajectory.times[START])
    slice_time_traj = trajectory.filter_by_time(trajectory.times[START],
        trajectory.times[STOP])

    assert single_time_traj.times == trajectory.times[START]
    assert np.all(slice_time_traj.times) == np.all(trajectory.times[START:STOP])
