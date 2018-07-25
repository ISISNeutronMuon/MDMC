"""System tests for total, coherent and incoherent SQw and FQt calculations from
MD

Although SQw and FQt are two separate observables, as the calculation of SQw
realies on the calculation of FQt they are tested together.

AUTHOR :    Thomas Farmer        START DATE :    24/07/2018, 15:34:26"""

import pytest
import numpy as np
from netCDF4 import Dataset

import MDMC.src.trajectory_analysis.observables.obs_factory as eof

from MDMC.tests.test_data import data

@pytest.fixture
def incoh_file():
    return Dataset(data.obs_data['SQw_incoh'],'r')

@pytest.fixture
def coh_file():
    return Dataset(data.obs_data['SQw_coh'],'r')

@pytest.fixture
def Q(incoh_file):
    return np.array(incoh_file.variables['q'][:])

@pytest.fixture
def time(incoh_file):
    return np.array(incoh_file.variables['time'][:])

@pytest.fixture
def w(incoh_file):
    return np.array(incoh_file.variables['frequency'][:])

@pytest.fixture
def FQt_incoh(incoh_file):
    return np.array(incoh_file.variables['Fqt-total'][:])

@pytest.fixture
def SQw_incoh(incoh_file):
    return np.array(incoh_file.variables['Sqw-total'][:])

@pytest.fixture
def FQt_coh(coh_file):
    return np.array(coh_file.variables['Fqt-total'][:])

@pytest.fixture
def SQw_coh(coh_file):
    return np.array(coh_file.variables['Sqw-total'][:])


def test_FQt_incoh(Q, time, FQt_incoh):

    """
    Validate the calculation of the intermediate incoherent structure factor
    against nMOLDYN
    """

    raise NotImplementedError


def test_FQt_coh(Q, time, FQt_coh):

    """
    Validate the calculation of the intermediate coherent structure factor
    against nMOLDYN
    """

    raise NotImplementedError


def test_FQt_total(Q, time, FQt_incoh, FQt_coh):

    """
    Validate the calculation of the intermediate total structure factor against
    the sum of the intermediate incoherent and coherent structure factors
    calculated by MOLDYN
    """

    raise NotImplementedError


def test_SQw_incoh(Q, w, SQw_incoh):

    """
    Validate the calculation of the dynamic incoherent structure factor against
    nMOLDYN
    """

    raise NotImplementedError


def test_SQw_coh(Q, w, SQw_coh):

    """
    Validate the calculation of the dynamic coherent structure factor against
    nMOLDYN
    """

    raise NotImplementedError


def test_SQw_total(Q, w, SQw_incoh, SQw_coh):

    """
    Validate the calculation of the dynamic total structure factor against the
    sum of the dynamic incoherent and coherent structure factors calculated by
    nMOLDYN
    """

    raise NotImplementedError
