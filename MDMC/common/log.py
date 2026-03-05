"""
This module configures logging for MDMC.
"""

import logging

LOG_NAME = "MDMC"

def start_logging(logfile: str = "MDMC.log",
                  level: int = logging.INFO,
                  capture_warnings: bool = True) -> logging.Logger:
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
    return logger


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


def create_logger(name_override: str = "",
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

    logger = logging.getLogger(name=name_override if name_override else LOG_NAME)
    logger.setLevel(level)

    # Setup log file handler
    logging_fh = logging.FileHandler(logfile, mode='w')
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging_fh.setFormatter(formatter)
    logger.addHandler(logging_fh)

    return logger


def change_logging_level(name_override: str = "",
                         level: int = logging.INFO):
    """Change the log level of the selected logger.

    Parameters
    ----------
    name_override : str, optional
        If not empty, this will replace the standard log name, by default ""
    level : int, optional
        New logging level, by default logging.INFO
    """
    logger = logging.getLogger(name=name_override if name_override else LOG_NAME)
    logger.setLevel(level)
