"""
This module configures logging for MDMC.
"""

import logging
from enum import IntEnum


class LogLevels(IntEnum):
    """
    IntEnum mapping of logging levels.
    """
    #: ``logging.NOTSET``
    NOTSET = 0
    #: ``logging.DEBUG``
    DEBUG = 10
    #: ``logging.INFO``
    INFO = 20
    #: ``logging.WARNING``
    WARNING = 30
    #: ``logging.ERROR``
    ERROR = 40
    #: ``logging.CRITICAL``
    CRITICAL = 50


def start_logging(logfile: str = "MDMC.log",
                  level: LogLevels = LogLevels.INFO,
                  capture_warnings: bool = True):
    """
    Start one or more loggers to capture log information from MDMC.

    Parameters
    ----------
    logfile : str, optional
        The base name of the logfile.
    level : LogLevels, optional
        The logging level, corresponding to values in standard library logging
        module.  Default is 20 (``logging.INFO``).
    capture_warnings : bool, optional
        Whether warnings are captured by the logger (with a level of
        ``WARNING``) or printed to stdout.
    """

    logger = _start_single_logger(logfile, level=level)
    if capture_warnings:
        _capture_warnings(logger)


def _start_single_logger(logfile: str, level: LogLevels) -> logging.Logger:
    """
    Start a single logger and write startup message.

    Parameters
    ----------
    logfile : str
        Filename to start logging in.
    level : LogLevels
        Level of logging to support.

    Returns
    -------
    logging.Logger
        Logger with given filename and logging level.
    """

    logger = create_logger(logfile=logfile, level=level)
    logger.info("MDMC started logging to %s", logfile)

    return logger


def _capture_warnings(logger: logging.Logger):
    """
    Modify a logger to capture warnings issued in MDMC.

    Parameters
    ----------
    logger : logging.Logger
        Logger to add warnings support.

    Notes
    -----
    :any:`logging` module only provides `warnings` capturing at a module level,
    rather as a `Logger` method, so warnings have to be captured by default
    module level `logger` ("py.warnings") which then has the file handler
    from the logger attached.
    """

    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    file_handler = list(filter(lambda x: isinstance(x, logging.StreamHandler),
                               logger.handlers))[0]
    warnings_logger.addHandler(file_handler)


def create_logger(name: str = "MDMC",
                  logfile: str = "MDMC.log",
                  level: LogLevels = logging.INFO) -> logging.Logger:
    """
    Create a formatter logger which outputs to a log file.

    Parameters
    ----------
    name : str
        The name of the logger.
    logfile : str
        The name of the log file.
    level : LogLevels
        The debug level of the logger.

    Returns
    -------
    logging.Logger
        Logger with given filename and logging level.
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
