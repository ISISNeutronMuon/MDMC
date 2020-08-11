"""This module configures logging for MDMC
"""

import logging
import platform
from typing import List, Union

from mpi4py import MPI


def start_logging(logfile: str = "MDMC.log",
                  level: int = logging.INFO,
                  ranks: Union[int, List[int]] = 0):

    """
    Parameters
    ----------
    logfile : str, optional
        The base name of the logfile
    level : int, optional
        The logging level, corresponding to values in standard library logging
        module.  Default is 20 (``logging.INFO``).
    ranks : int, list, optional
        An `int` or `list` of `int` which specifies each rank which will log.
        Each of these ranks will produce a separate log file, which will be the
        base ``logfile`` string and the node name and rank. `-1` indicates all
        ranks will be logged. **It is recommended `-1` is not used for runs
        using a large number of ranks.**
    """

    if ranks:
        if ranks == -1:
            ranks = range(0, MPI.COMM_WORLD.Get_size(), 1)
        rank = MPI.COMM_WORLD.rank
        if rank in ranks:
            # Prepends rank in front of .log extension if it exists, otherwise
            # appends to logfile
            add = '_{0}_{1}'.format(platform.node(), rank)
            logfile = ('{0}{1}'.format(logfile, add)).replace(
                '.log{}'.format(add), '{}.log'.format(add))
            logger = create_logger(logfile=logfile, level=level)
            logger.info("MDMC started logging to %s", logfile)
    else:
        _start_single_logger(logfile)


def stop_logging():

    pass


def _start_single_logger(logfile):

    logger = create_logger(logfile=logfile)
    logger.info("MDMC started logging to %s", logfile)


def create_logger(name: str = "MDMC",
                  logfile: str = "MDMC.log",
                  level: int = logging.INFO) -> logging.Logger:

    logger = logging.getLogger(name=name)
    logger.setLevel(level)

    # Setup log file handler
    logging_fh = logging.FileHandler(logfile, mode='w')
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging_fh.setFormatter(formatter)
    logger.addHandler(logging_fh)

    return logger
