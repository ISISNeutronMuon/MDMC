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

"""Tools for concurrency in observable calculation."""

import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from typing import TypeVar

T = TypeVar("T")


def create_executor() -> ThreadPoolExecutor:
    """
    Create a ``ThreadPoolExecutor`` with the relevant number of workers.

    Returns
    -------
    ThreadPoolExecutor
        A thread pool executor with max_workers=`OMP_NUM_THREADS` or 1 if not set.
    """

    # we use a ThreadPoolExecutor as most of the concurrent operations
    # involve very large arrays; a ProcessPoolExecutor would create a
    # copy of each of these arrays per thread.
    num_cores = int(os.environ.get("OMP_NUM_THREADS", 1))
    return ThreadPoolExecutor(max_workers=num_cores)


def core_batch(generator: Iterable[T]) -> Iterable[tuple[T, ...]]:
    """
    Batch generator according to the number of available cores, `OMP_NUM_THREADS`.

    Parameters
    ----------
    generator : Iterable[T]
        The generator to batch.

    Yields
    ------
    tuple[T]
        Batches of size `OMP_NUM_THREADS`.

    See Also
    --------
    itertools.batched : Standard implementation from 3.12.

    Examples
    --------

        >>> core_batch(range(10))
        on 1 core produces [0], [1], [2], [3], [4], [5], [6], [7], [8], [9]
        on 4 cores produces [0, 1, 2, 3], [4, 5, 6, 7], [8, 9].
    """
    num_cores = int(os.environ.get("OMP_NUM_THREADS", 1))

    iterator = iter(generator)
    while batch := tuple(islice(iterator, num_cores)):
        yield batch
