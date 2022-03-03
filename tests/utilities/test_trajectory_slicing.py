"""Tests the slice_trajectory method
"""
import numpy as np
import pytest

from MDMC.utilities.trajectory_slicing import slice_trajectory
from tests.system_tests.observables.data_manager import trajectory


@pytest.mark.parametrize("subtrj_len, cont_slicing, expected_ranges",
             [(25, False, [(0, 24), (25, 49)]),
              (24, False, [(2, 25), (26, 49)]),
              (26, False, [(24, 49)]),
              (48, True, [(0, 47), (1, 48), (2, 49)])])
def test_slice_trajectory(subtrj_len, cont_slicing, expected_ranges, trajectory):
    actual_slices = slice_trajectory(trajectory, subtrj_len, cont_slicing)
    expected_slices = []
    for range in expected_ranges:
        expected_slices.append(trajectory[range[0]:range[1]])
    for i, slice in enumerate(actual_slices):
        assert np.all(expected_slices[i].times == slice.times)
        assert np.all((expected_slices[i].positions == slice.positions))
        assert np.all((expected_slices[i].velocities == slice.velocities))
        assert np.all((expected_slices[i].data == slice.data))
