"""Utility for slicing a ``Trajectory`` object into sub-trajectories"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterable
    from MDMC.trajectory_analysis.trajectory import Trajectory

# the type hint for trj should be trj: Trajectory, but importing `Trajectory` would currently
# lead to a circular import


def slice_trajectory(trj: "Trajectory", subtrj_len: int, cont_slicing: bool = False) \
        -> "Iterable[Trajectory]":
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
            [Trajectory[0:5], Trajectory[5:10]]
        slice_trajectory(Trajectory, subtrj_len=4, cont_slicing=False):
            [Trajectory[2:6], Trajectory[6:10]]
        slice_trajectory(Trajectory, subtrj_len=8, cont_slicing=True):
            [Trajectory[0:8], Trajectory[1:9],Trajectory[2:10]]

    Returns
    -------
    Iterable[Trajectory]
        A generator function for the list of sub-trajectories of length ``subtrj_len``.
    """
    trj_len = len(trj)

    msg = (f'The sub-trajectory length of {subtrj_len} was larger than the length of the '
           f'parent trajectory of {trj_len}.')
    assert trj_len >= subtrj_len, msg

    if cont_slicing:
        first_frame = 0
        slice_step = 1
    else:
        first_frame = trj_len % subtrj_len
        slice_step = subtrj_len

    for i in range(first_frame, trj_len-subtrj_len+1, slice_step):
        yield trj[i: i+subtrj_len]
