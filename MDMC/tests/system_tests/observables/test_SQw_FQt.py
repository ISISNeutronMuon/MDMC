"""System tests for total, coherent and incoherent SQw and FQt calculations from
MD

Although SQw and FQt are two separate observables, as the calculation of SQw
realies on the calculation of FQt they are tested together.

AUTHOR :    Thomas Farmer        START DATE :    24/07/2018, 15:34:26"""

try:
    import cPickle as pickle
except:
    import pickle
import zlib

from netCDF4 import Dataset
import numpy as np
import pytest

import MDMC.src.trajectory_analysis.observables.obs_factory as of

from MDMC.tests.test_data import data

# Values are equivalent to those used by nMOLDYN to generate the test data
START = 50
STOP = 5010
STEP = 100
n_Q = 13
CELL = (3.94221067, 3.94221067, 3.94221067)

@pytest.fixture
def incoh_file():
    return Dataset(data.OBS_DATA['SQw_incoh'],'r')

@pytest.fixture
def coh_file():
    return Dataset(data.OBS_DATA['SQw_coh'],'r')

@pytest.fixture
def Q_ref(incoh_file):
    return np.array(incoh_file.variables['q'][:])

@pytest.fixture
def time_ref(incoh_file):
    return np.array(incoh_file.variables['time'][:])

@pytest.fixture
def w_ref(incoh_file):
    return np.array(incoh_file.variables['frequency'][:])

@pytest.fixture
def FQt_incoh_ref(incoh_file):
    return np.array(incoh_file.variables['Fqt-total'][:])

@pytest.fixture
def SQw_incoh_ref(incoh_file):
    return np.array(incoh_file.variables['Sqw-total'][:])

@pytest.fixture
def FQt_coh_ref(coh_file):
    return np.array(coh_file.variables['Fqt-total'][:])

@pytest.fixture
def SQw_coh_ref(coh_file):
    return np.array(coh_file.variables['Sqw-total'][:])

@pytest.fixture
def SQw_obs():

    """
    Setup the container for Q, time, w, FQt and SQt

    trajectory is unzipped and unpickled. Q_values are rounded to ensure
    consistency, as nMOLDYN rounds.
    """

    compressed_trajectory = open(data.OBJECT_DATA['trajectory'], 'r').read()
    pickled_trajectory = zlib.decompress(compressed_trajectory)
    trajectory = pickle.loads(pickled_trajectory)
    Q_values = np.around([2 * np.pi * i / CELL[0] for i in range(1, n_Q+1)], 1)
    SQw = of.ObservableFactory.create_observable('SQw')
    SQw.calculate_from_MD(trajectory, Q_values = Q_values, cell = CELL)


def test_Q(Q_ref, SQw_obs):

    """
    Test Q equivalence
    """

    raise NotImplementedError


def test_time(time_ref, SQw_obs):

    """
    Test time equivalence
    """

    raise NotImplementedError


def test_w(w_ref, SQw_obs):

    """
    Test frequency equivalence
    """

    raise NotImplementedError


def test_FQt_incoh(FQt_incoh_ref, SQw_obs):

    """
    Validate the calculation of the intermediate incoherent structure factor
    against nMOLDYN
    """

    raise NotImplementedError


def test_FQt_coh(FQt_coh_ref, SQw_obs):

    """
    Validate the calculation of the intermediate coherent structure factor
    against nMOLDYN
    """

    raise NotImplementedError


def test_FQt_total(FQt_incoh_ref, FQt_coh_ref, SQw_obs):

    """
    Validate the calculation of the intermediate total structure factor against
    the sum of the intermediate incoherent and coherent structure factors
    calculated by MOLDYN
    """

    raise NotImplementedError


def test_SQw_incoh(SQw_incoh_ref, SQw_obs):

    """
    Validate the calculation of the dynamic incoherent structure factor against
    nMOLDYN
    """

    raise NotImplementedError


def test_SQw_coh(SQw_coh_ref, SQw_obs):

    """
    Validate the calculation of the dynamic coherent structure factor against
    nMOLDYN
    """

    raise NotImplementedError


def test_SQw_total(SQw_incoh_ref, SQw_coh_ref, SQw_obs):

    """
    Validate the calculation of the dynamic total structure factor against the
    sum of the dynamic incoherent and coherent structure factors calculated by
    nMOLDYN
    """

    raise NotImplementedError
