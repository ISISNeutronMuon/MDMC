"""Tools for concurrency in observable calculation."""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Generator, List


def create_executor() -> ThreadPoolExecutor:
    """
    Creates a ThreadPoolExecutor with the
    relevant number of workers (according to
    the number of cores specified).

    Returns
    -------
    ThreadPoolExecutor
        A thread pool executor with max_workers=`OMP_NUM_THREADS`
    """

    # we use a ThreadPoolExecutor as most of the concurrent operations
    # involve very large arrays; a ProcessPoolExecutor would create a 
    # copy of each of these arrays per thread.
    num_cores = int(os.environ.get("OMP_NUM_THREADS", 1))
    return ThreadPoolExecutor(max_workers=num_cores)


def core_batch(generator: Generator) -> Generator[List, None, None]:
    """
    Turn a generator into a new generator that yields in batches according
    to the number of available cores, `OMP_NUM_THREADS`.

    Example:
    >>> core_batch((i for i in range (10))
    on 1 core produces [0], [1], [2], [3], [4], [5], [6], [7], [8], [9]
    on 4 cores produces [0, 1, 2, 3], [4, 5, 6, 7], [8, 9].

    Parameters
    ----------
    generator: Generator
        The generator to batch.

    Returns
    -------
    Generator
        A generator which iterates in batches of size `OMP_NUM_THREADS`.
    """

    num_cores = int(os.environ.get("OMP_NUM_THREADS", 1))

    generator_not_exhausted = True

    while generator_not_exhausted:
        batch = []
        for _ in range(num_cores):
            try:
                batch.append(next(generator))
            except StopIteration:
                generator_not_exhausted = False
                break
        yield batch
