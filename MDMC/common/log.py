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

"""
This module configures logging for MDMC.
"""

import logging


def start_logging(logfile: str = "MDMC.log",
                  level: int = logging.INFO,
                  capture_warnings: bool = True):
    """
    Start one or more loggers to capture log information from MDMC.

    Parameters
    ----------
    logfile : str, optional
        The base name of the logfile.
    level : int, optional
        The logging level, corresponding to values in standard library logging
        module.
    capture_warnings : bool, optional
        Whether warnings are captured by the logger (with a level of
        WARNING) or printed to stdout.
    """

    logger = _start_single_logger(logfile, level=level)
    if capture_warnings:
        _capture_warnings(logger)


def _start_single_logger(logfile: str, level: int) -> logging.Logger:
    """
    Start a single MDMC logger.

    Parameters
    ----------
    logfile : str
        Path for log to write to.
    level : int
        Level of logger to start.

    Returns
    -------
    logging.Logger
        Single MDMC logger.
    """

    logger = create_logger(logfile=logfile, level=level)
    logger.info("MDMC started logging to %s", logfile)

    return logger


def _capture_warnings(logger: logging.Logger):
    """
    Enable warning capture for `logger`.

    Parameters
    ----------
    logger : logging.Logger
        Logger on which to enable warnings.

    Notes
    -----
    Logging module only provides warnings capturing at a module level,
    rather as a Logger method, so warnings have to be captured by default
    module level logger ("py.warnings") which then has the file handler
    from the logger attached.
    """

    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    file_handler = list(filter(lambda x: isinstance(x, logging.StreamHandler),
                               logger.handlers))[0]
    warnings_logger.addHandler(file_handler)


def create_logger(name: str = "MDMC",
                  logfile: str = "MDMC.log",
                  level: int = logging.INFO) -> logging.Logger:
    """
    Create a formatter logger which outputs to a log file.

    Parameters
    ----------
    name : str, optional
        The name of the logger.
    logfile : str, optional
        The name of the log file.
    level : int, optional
        The debug level of the logger.

    Returns
    -------
    logging.Logger
        Logger to handle MDMC logging.
    """

    logger = logging.getLogger(name=name)
    logger.setLevel(level)

    # Setup log file handler
    logging_fh = logging.FileHandler(logfile, mode='w')
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging_fh.setFormatter(formatter)
    logger.addHandler(logging_fh)

    return logger
