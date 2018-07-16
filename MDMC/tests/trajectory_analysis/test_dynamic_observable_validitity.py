"""This module will validate the calculation of each dynamic observable by
comparing the output with data calculated from other software.

AUTHOR :    Thomas Farmer        START DATE :    16/07/2018, 11:26:03"""

import pytest

from MDMC.tests.test_data import data


def test_FQt_validation():

    """
    Validates the FQt calculation
    """

    raise NotImplementedError

def test_SQw_validation():

    """
    Validates the SQw calculation

    This is validated against both external data and calculating SQw using the
    longitudinal component of the particle current
    """

    raise NotImplementedError
