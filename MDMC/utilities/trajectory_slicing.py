# MDMC is a package for the optimisation of classical potentials with experimental data
# Copyright (C) 2026 MDMC Developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Utility for slicing a ``CompactTrajectory`` object into sub-trajectories"""

from collections.abc import Iterable

from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory


def slice_trajectory(
        trj: CompactTrajectory,
        subtrj_len: int,
        cont_slicing: bool = False,
) -> Iterable[CompactTrajectory]:
    """
    Takes a ``CompactTrajectory`` object and slices it into a list of shorter
    ``CompactTrajectory`` objects of equal length.

    Parameters
    ----------
    trj : CompactTrajectory
        The trajectory that should be divided into smaller sub-trajectories.
    subtrj_len : int
        The length (number of frames) of the sub-trajectories.
    cont_slicing : bool, optional
        Flag to set if a rolling/continuous slicing should be used with frames allowed to
        appear in multiple sub-trajectories.

        If set to ``False`` (default) then the sub-trajectories are separate
        subsets of the ``CompactTrajectory`` with distinct frames.

        Note that if set to ``False`` it checks if ``len(trj)%subtrj_len==0`` and if not it does
        not use the first ``len(trj)%subtrj_len`` frames of the ``CompactTrajectory``.

    Yields
    ------
    CompactTrajectory
        Sub-trajectories of length ``subtrj_len``.

    Examples
    --------
    If ``len(CompactTrajectory)==10`` then the following examples would give:

    >>> slice_trajectory(CompactTrajectory, subtrj_len=5, cont_slicing=False):
    [CompactTrajectory[0:5], CompactTrajectory[5:10]]
    >>> slice_trajectory(CompactTrajectory, subtrj_len=4, cont_slicing=False):
    [CompactTrajectory[2:6], CompactTrajectory[6:10]]
    >>> slice_trajectory(CompactTrajectory, subtrj_len=8, cont_slicing=True):
    [CompactTrajectory[0:8], CompactTrajectory[1:9], CompactTrajectory[2:10]]
    """
    trj_len = len(trj)

    if trj_len < subtrj_len:
        raise IndexError(f'The sub-trajectory length of {subtrj_len} was larger than '
                         f'the length of the parent trajectory of {trj_len}.')

    if cont_slicing:
        first_frame = 0
        slice_step = 1
    else:
        first_frame = trj_len % subtrj_len
        slice_step = subtrj_len

    for i in range(first_frame, trj_len-subtrj_len+1, slice_step):
        yield trj.subtrajectory(i, i+subtrj_len)
