"""Tests for SQw observable

Includes calculation from MD trajectory and reading from experimental data file.

AUTHOR :    Thomas Farmer        START DATE :    2018-5-29 16:34:02"""

import pytest
import numpy as np

import MDMC.src.trajectory_analysis.observables.obs_factory as of

from MDMC.tests.test_data import data
from MDMC.tests.trajectory_analysis.test_histogram import trajectory
from MDMC.tests.MD.test_simulation import water_SPCE_universe, water_molecule, \
    atom, universe

@pytest.fixture
def SQw_from_data():
    SQw = of.ObservableFactory.create_observable('SQw')
    SQw.read_from_file(reader='LAMPSQw', file_name=data.READER_DATA['LAMPSQw'])
    return SQw

@pytest.fixture
def SQw_from_MD(trajectory, universe):
    SQw = of.ObservableFactory.create_observable('SQw')
    cell = universe.dims
    n_Q = 10
    Q_values = [2 * np.pi * i / cell[0] for i in range(1,n_Q+1)]
    SQw.calculate_from_MD(trajectory, Q_values = Q_values, cell = cell)
    return SQw

# TODO: Test for consistency by comparing S(Q,w) where w = 0 with S(Q)
# TODO: Add plain text files with expected values of E and SQw and then compare with parsed values


def test_from_data(SQw_from_data):

    """
    Test the following:

    - _from_MD flag is False
    - reader is LAMPSQw
    - Q and E are the independent variables
    - SQw is the dependent variable
    - SQw is the variable on which there is an error
    - Q ranges from 0 to 3.5 in 0.05 increments
    """

    assert SQw_from_data.origin == 'experiment'
    assert SQw_from_data.reader.__class__.__name__ == "LAMPSQw"

    assert 'Q' in SQw_from_data.independent_variables and \
        'E' in SQw_from_data.independent_variables
    assert 'SQw' in SQw_from_data.dependent_variables
    assert 'SQw' in SQw_from_data.errors

    assert np.all(SQw_from_data.independent_variables['Q']) == \
        np.all(np.arange(0, 3.55, 0.05))


def test_from_MD(SQw_from_MD):

    """
    Test the following:


    """

    pass
