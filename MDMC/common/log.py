"""This module configures logging for MDMC"""

import logging


def start_logging(logfile: str = "MDMC.log",
                  level: int = logging.INFO,
                  capture_warnings: bool = True):
    """
    Start one or more loggers to capture log information from MDMC

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
    capture_warnings : bool, optional
        Whether warnings are captured by the logger (with a level of
        WARNING) or printed to stdout.
    """

    logger = _start_single_logger(logfile, level=level)
    if capture_warnings:
         _capture_warnings(logger)


def _start_single_logger(logfile: str, level: int) -> logging.Logger:

    logger = create_logger(logfile=logfile, level=level)
    logger.info("MDMC started logging to %s", logfile)

    return logger


def _capture_warnings(logger: logging.Logger):

    # logging module only provides warnings capturing at a module level,
    # rather as a Logger method, so warnings have to be captured by default
    # module level logger ("py.warnings") which then has the file handler
    # from the logger attached.
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    file_handler = list(filter(lambda x: isinstance(x, logging.StreamHandler),
                               logger.handlers))[0]
    warnings_logger.addHandler(file_handler)


def create_logger(name: str = "MDMC",
                  logfile: str = "MDMC.log",
                  level: int = logging.INFO) -> logging.Logger:
    """
    Create a formatter logger which outputs to a log file

    Parameters
    ----------
    name : str
        The name of the logger
    logfile : str
        The name of the log file
    level : int
        The debug level of the logger
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
