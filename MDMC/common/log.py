"""This module configures logging for MDMC
"""

import logging


def start_logging(logfile: str = "MDMC.log"):

    logger = create_logger(logfile=logfile)
    logger.info("MDMC started logging to %s", logfile)


def create_logger(name: str = "MDMC",
                  logfile: str = "MDMC.log",
                  level: int = logging.DEBUG) -> logging.Logger:

    logger = logging.getLogger(name=name)
    logger.setLevel(level)

    # Setup log file handler
    logging_fh = logging.FileHandler(logfile, mode='w')
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging_fh.setFormatter(formatter)
    logger.addHandler(logging_fh)

    return logger
