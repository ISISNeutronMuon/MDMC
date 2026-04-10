"""Tools for concurrency in observable calculation."""

import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from typing import TypeVar

T = TypeVar('T')


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
