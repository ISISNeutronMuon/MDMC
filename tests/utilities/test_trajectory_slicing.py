"""Tests the slice_trajectory method
"""
import numpy as np
from MDMC.utilities.trajectory_slicing import slice_trajectory
from tests.system_tests.observables.data_manager import trajectory

def test_slice_trajectory(trajectory):
    # the loaded trajectory is a pytest fixture and as such cannot be used in pytest's
    # parametrize method, so we have to parametrize the tests in a dumber way
    trj = trajectory
    tests_to_run = ((25, False, [trj[0:24], trj[25:49]]),
                    (24, False, [trj[2:25], trj[26:49]]),
                    (26, False, [trj[24:49]]),
                    (48, True, [trj[0:47], trj[1:48], trj[2:49]]))
    for subtrj_len, cont_slicing, expected_slices in tests_to_run:
        actual_slices = slice_trajectory(trj, subtrj_len, cont_slicing)
        # we cannot compare the subtrajectories directly as ``Trajectory`` has no __eq__
        # method. instead we compare the content of the subtrajectories
        for i in range(len(expected_slices)):
            assert np.all(expected_slices[i].times == actual_slices[i].times)
            assert np.all((expected_slices[i].positions == actual_slices[i].positions))
            assert np.all((expected_slices[i].velocities == actual_slices[i].velocities))
            assert np.all((expected_slices[i].data == actual_slices[i].data))

