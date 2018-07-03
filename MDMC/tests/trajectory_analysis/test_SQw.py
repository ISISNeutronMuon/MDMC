"""Tests for the calculation of Sqw from MD trajectory output

AUTHOR :    Thomas Farmer        START DATE :    2018-5-29 16:34:02"""

import pytest

from MDMC.tests.test_data import data
from MDMC.tests.trajectory_analysis.test_histogram import trajectory

import MDMC.src.trajectory_analysis.observables.exp_obs_factory as eof

@pytest.fixture
def SQw_from_data():
    SQw = eof.ExperimentalObservableFactory.create_observable('SQw')
    SQw.read_from_file(reader='LAMPSQw', file_name=data.LAMP_SQW_FILE)
    return SQw.data

@pytest.fixture
def SQw_from_MD(trajectory):
    SQw = eof.ExperimentalObservableFactory.create_observable('SQw')
    params = {}
    SQw.calculate_from_MD(trajectory, params)


# TODO: Test for  consistency by comparing S(Q,w) where w = 0 with S(Q)

def test_from_data(SQw_from_data):
    pass
