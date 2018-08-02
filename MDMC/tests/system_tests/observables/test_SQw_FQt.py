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
from numpy.testing import assert_allclose
import pytest

import MDMC.src.trajectory_analysis.observables.obs_factory as of

from MDMC.tests.test_data import data

# Values are equivalent to those used by nMOLDYN to generate the test data
CELL = (3.94221067, 3.94221067, 3.94221067)
T_RESOLUTION = 30.999425

# As all FQt should be normalised to 1, the absolute tolerance for
# comparisons with reference data is set.  As SQw data has been FFT, the effects
# if noise are less significant, so relative tolerance can be used.
ATOL = 0.015
RTOL = 0.050

@pytest.fixture(scope="module")
def incoh_file():
    return Dataset(data.OBS_DATA['SQw_incoh'],'r')

@pytest.fixture(scope="module")
def coh_file():
    return Dataset(data.OBS_DATA['SQw_coh'],'r')

@pytest.fixture(scope="module")
def Q_ref(incoh_file):
    return np.array(incoh_file.variables['q'][:])

@pytest.fixture(scope="module")
def time_ref(incoh_file):
    return np.array(incoh_file.variables['time'][:])

@pytest.fixture(scope="module")
def w_ref(incoh_file):
    return np.array(incoh_file.variables['angular_frequency'][:])

@pytest.fixture(scope="module")
def FQt_incoh_ref(incoh_file):
    return np.array(incoh_file.variables['Fqt-total'][:])

@pytest.fixture(scope="module")
def SQw_incoh_ref(incoh_file):
    return np.array(incoh_file.variables['Sqw-total'][:])

@pytest.fixture(scope="module")
def FQt_coh_ref(coh_file):
    return np.array(coh_file.variables['Fqt-total'][:])

@pytest.fixture(scope="module")
def SQw_coh_ref(coh_file):
    return np.array(coh_file.variables['Sqw-total'][:])

@pytest.fixture(scope="module")
def trajectory():

    """
    Read the trajectory

    trajectory is unzipped and unpickled.
    """

    compressed_trajectory = open(data.OBJECT_DATA['trajectory'], 'r').read()
    pickled_trajectory = zlib.decompress(compressed_trajectory)
    trajectory = pickle.loads(pickled_trajectory)
    return trajectory

@pytest.fixture(scope="module")
def Q_values():

    return np.arange(1.6, 21, 1.6)

@pytest.fixture(scope="module")
def SQw_obs(trajectory, Q_values):

    """
    Setup the container for Q, time, w, total FQt and total SQt
    """

    SQw = of.ObservableFactory.create_observable('SQw')
    SQw.calculate_from_MD(trajectory, Q_values=Q_values, cell=CELL,
                          t_resolution=T_RESOLUTION)
    return SQw

@pytest.fixture(scope="module")
def SQw_incoh_obs(trajectory, Q_values):

    """
    Setup the container for Q, time, w, incoherent FQt and incoherent SQt

    Only FQt and SQt are used for testing, as Q, time and w are calculated
    using the same base class as SQw_obs
    """

    SQw_incoh = of.ObservableFactory.create_observable('SQw_incoh')
    SQw_incoh.calculate_from_MD(trajectory, Q_values=Q_values, cell=CELL,
                                t_resolution=T_RESOLUTION)
    return SQw_incoh

@pytest.fixture(scope="module")
def SQw_coh_obs(trajectory, Q_values):

    """
    Setup the container for Q, time, w, coherent FQt and coherent SQt

    Only FQt and SQt are used for testing, as Q, time and w are calculated
    using the same base class as SQw_obs
    """

    SQw_coh = of.ObservableFactory.create_observable('SQw_coh')
    SQw_coh.calculate_from_MD(trajectory, Q_values=Q_values, cell=CELL,
                              t_resolution=T_RESOLUTION)
    return SQw_coh


def test_Q(Q_ref, SQw_obs):

    """
    Test Q equivalence

    Exact equivalence results in failed assertion due to rounding errors caused
    by the range routine used in nMOLDYN
    """

    assert_allclose(SQw_obs.Q_values, Q_ref, atol=1e-07)


def test_time(time_ref, SQw_obs):

    """
    Test time equivalence
    """

    assert np.all(SQw_obs.t == time_ref)


def test_w(w_ref, SQw_obs):

    """
    Test angular frequency equivalence

    Exact equivalence results in failed assertion due to rounding errors
    """

    assert_allclose(SQw_obs.w, w_ref, atol=1e-07)


def test_FQt_incoh(FQt_incoh_ref, SQw_incoh_obs):

    """
    Validate the calculation of the intermediate incoherent structure factor
    against nMOLDYN
    """

    assert np.all(np.shape(SQw_incoh_obs.FQt) == np.shape(FQt_incoh_ref))
    assert_allclose(SQw_incoh_obs.FQt, FQt_incoh_ref, atol=ATOL)


def test_FQt_coh(FQt_coh_ref, SQw_coh_obs):

    """
    Validate the calculation of the intermediate coherent structure factor
    against nMOLDYN
    """

    assert np.all(np.shape(SQw_coh_obs.FQt) == np.shape(FQt_coh_ref))
    assert_allclose(SQw_coh_obs.FQt, FQt_coh_ref, atol=ATOL)


def test_FQt_total(FQt_incoh_ref, FQt_coh_ref, SQw_obs):

    """
    Validate the calculation of the intermediate total structure factor against
    the sum of the intermediate incoherent and coherent structure factors
    calculated by MOLDYN
    """

    assert np.all(np.shape(SQw_obs.FQt) == np.shape(FQt_incoh_ref))
    FQt_ref = FQt_incoh_ref + FQt_coh_ref
    assert_allclose(SQw_obs.FQt, FQt_ref, atol=ATOL)


def test_SQw_incoh(SQw_incoh_ref, SQw_incoh_obs):

    """
    Validate the calculation of the dynamic incoherent structure factor against
    nMOLDYN
    """

    assert np.all(np.shape(SQw_incoh_obs.SQw) == np.shape(SQw_incoh_ref))
    assert_allclose(SQw_incoh_obs.SQw, SQw_incoh_ref, rtol=RTOL)


def test_SQw_coh(SQw_coh_ref, SQw_coh_obs):

    """
    Validate the calculation of the dynamic coherent structure factor against
    nMOLDYN
    """

    assert np.all(np.shape(SQw_coh_obs.SQw) == np.shape(SQw_coh_ref))
    assert_allclose(SQw_coh_obs.SQw, SQw_coh_ref, rtol=RTOL)


def test_SQw_total(SQw_incoh_ref, SQw_coh_ref, SQw_obs):

    """
    Validate the calculation of the dynamic total structure factor against the
    sum of the dynamic incoherent and coherent structure factors calculated by
    nMOLDYN
    """

    assert np.all(np.shape(SQw_obs.SQw) == np.shape(SQw_incoh_ref))
    SQw_ref = SQw_incoh_ref + SQw_coh_ref
    assert_allclose(SQw_obs.SQw, SQw_ref, rtol=RTOL)
