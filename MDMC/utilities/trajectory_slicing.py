"""Utility for slicing a ``Trajectory`` object into sub-trajectories"""

import numpy as np
from typing import List
from MDMC.trajectory_analysis.trajectory import Trajectory


def slice_trajectory(trj: Trajectory, subtrj_len: int, cont_slicing: bool = False) -> List[
    Trajectory]:
    """
    Takes a ``Trajectory`` objects and slices it into a list of shorter ``Trajectory`` objects
    of the same length.

    Parameters
    ----------
    trj : Trajectory
        The trajectory that should be divided into smaller sub-trajectories.
    subtrj_len : int
        The length (number of frames) of the sub-trajectories.
    cont_slicing : bool
        Flag to set if a rolling/continuous slicing should be used with frames allowed to
        appear in multiple sub-trajectories. If set to ``False`` (default) then the
        sub-trajectories are separate subsets of the ``Trajectory`` with distinct frames.
        Note that if set to ``False`` it checks if ``len(trj)%subtrj_len==0`` and if not it does
        not use the first ``len(trj)%subtrj_len`` frames of the ``Trajectory``.

    Example
    -------
    If ``len(Trajectory)==10`` then the following examples would give:
        slice_trajectory(Trajectory, subtrj_len=5, cont_slicing=False):
            [Trajectory[0:4], Trajectory[5:9]]
        slice_trajectory(Trajectory, subtrj_len=4, cont_slicing=False):
            [Trajectory[2:5], Trajectory[5:9]]
        slice_trajectory(Trajectory, subtrj_len=8, cont_slicing=True):
            [Trajectory[0:7], Trajectory[1:8],Trajectory[2:9]]

    Returns
    -------
    List[Trajectory]
        A list of sub-trajectories of the same length.
    """
    subtrj_list = []
    trj_len = len(trj)
    if cont_slicing:
        first_frame = 0
        slice_step = 1
    else:
        first_frame = trj_len % subtrj_len
        slice_step = subtrj_len

    for i in range(first_frame, trj_len-subtrj_len+1, slice_step):
        subtrj_list.append(trj[i: i+subtrj_len-1])

    return subtrj_list
