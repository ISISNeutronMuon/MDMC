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

import MDMC.common.atom_properties as ap
import MDMC.trajectory_analysis.observables.obs_factory as of

from tests.test_data import data

# Values are equivalent to those used by nMOLDYN to generate the test data
DIMS = (39.4221067, 39.4221067, 39.4221067)
T_RESOLUTION = 30.999425

# Absolute tolerance is included to account for rounding differences in nMOLDYN
# and MDMC
ATOL = 1e-7

# Constants for correct normalisation relative to nMOLDYN FQt and SQw
N_H = 4096
N_O = 2048
N_TOTAL = N_H + N_O
N_H_O = np.sqrt(N_H * N_O)
B_FACTOR = (ap.B_INCOH['H']**2 * N_H + ap.B_INCOH['O']**2 * N_O) / N_TOTAL
N_Q_VALUES = 13

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
def FQt_coh_HH_ref(coh_file):
    return np.array(coh_file.variables['Fqt-HH'][:])

@pytest.fixture(scope="module")
def FQt_coh_HO_ref(coh_file):
    return np.array(coh_file.variables['Fqt-HO'][:])

@pytest.fixture(scope="module")
def FQt_coh_OO_ref(coh_file):
    return np.array(coh_file.variables['Fqt-OO'][:])

@pytest.fixture(scope="module")
def FQt_coh_ref(FQt_coh_HH_ref, FQt_coh_HO_ref, FQt_coh_OO_ref):
    FQt_coh_ref = (FQt_coh_HH_ref * ap.B_COH['H']**2 * N_H
                   + FQt_coh_HO_ref * ap.B_COH['H'] * ap.B_COH['O'] * N_H_O
                   + FQt_coh_OO_ref * ap.B_COH['O']**2 * N_O) / N_TOTAL
    return FQt_coh_ref

@pytest.fixture(scope="module")
def SQw_coh_HH_ref(coh_file):
    return np.array(coh_file.variables['Sqw-HH'][:])

@pytest.fixture(scope="module")
def SQw_coh_HO_ref(coh_file):
    return np.array(coh_file.variables['Sqw-HO'][:])

@pytest.fixture(scope="module")
def SQw_coh_OO_ref(coh_file):
    return np.array(coh_file.variables['Sqw-OO'][:])

@pytest.fixture(scope="module")
def SQw_coh_ref(SQw_coh_HH_ref, SQw_coh_HO_ref, SQw_coh_OO_ref):
    SQw_coh_ref = (SQw_coh_HH_ref * ap.B_COH['H']**2 * N_H
                   + SQw_coh_HO_ref * ap.B_COH['H'] * ap.B_COH['O'] * N_H_O
                   + SQw_coh_OO_ref * ap.B_COH['O']**2 * N_O) / N_TOTAL
    return SQw_coh_ref

@pytest.fixture(scope="module")
def trajectory():

    """
    Read the trajectory

    Trajectory is unzipped and unpickled.
    """

    compressed_trajectory = open(data.OBJECT_DATA['trajectory'], 'r').read()
    pickled_trajectory = zlib.decompress(compressed_trajectory)
    trajectory = pickle.loads(pickled_trajectory)
    return trajectory

@pytest.fixture(scope="module")
def Q_vectors():

    """
    Returns:
    An array of arrays of Q vectors for each Q value

    As Q vector calculations make a random selection of Q vectors from the set
    of all valid Q vectors, the Q vectors are set to the same values as those
    used in nMOLDYN when generating the incoherent FQt and SQw
    """

    return pickle.load(open(data.OBS_DATA['Q_vectors'], 'r'))

@pytest.fixture(scope="module")
def SQw_obs(trajectory, Q_vectors):

    """
    Setup the container for Q, time, w, total FQt and total SQt
    """

    SQw = of.ObservableFactory.create_observable('SQw')
    SQw.calculate_from_MD(trajectory, Q_vectors=Q_vectors, dims=DIMS,
                          t_resolution=T_RESOLUTION)
    return SQw

@pytest.fixture(scope="module")
def SQw_incoh_obs(trajectory, Q_vectors):

    """
    Setup the container for Q, time, w, incoherent FQt and incoherent SQt

    Only FQt and SQt are used for testing, as Q, time and w are calculated
    using the same base class as SQw_obs
    """

    SQw_incoh = of.ObservableFactory.create_observable('SQw_incoh')
    SQw_incoh.calculate_from_MD(trajectory, Q_vectors=Q_vectors, dims=DIMS,
                                t_resolution=T_RESOLUTION)
    return SQw_incoh

@pytest.fixture(scope="module")
def SQw_coh_obs(trajectory, Q_vectors):

    """
    Setup the container for Q, time, w, coherent FQt and coherent SQt

    Only FQt and SQt are used for testing, as Q, time and w are calculated
    using the same base class as SQw_obs
    """

    SQw_coh = of.ObservableFactory.create_observable('SQw_coh')
    SQw_coh.calculate_from_MD(trajectory, Q_vectors=Q_vectors, dims=DIMS,
                              t_resolution=T_RESOLUTION)
    return SQw_coh


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

    nMOLDYN normalises all FQt to 1, rather than the incoherent scattering cross
    section, so this factor is included.
    """

    assert np.all(np.shape(SQw_incoh_obs.FQt) == np.shape(FQt_incoh_ref))
    assert_allclose(SQw_incoh_obs.FQt / B_FACTOR, FQt_incoh_ref, atol=ATOL)


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
    # Coherent reference is already normalised - do the same for incoherent
    FQt_ref = FQt_incoh_ref * B_FACTOR + FQt_coh_ref
    assert_allclose(SQw_obs.FQt, FQt_ref, atol=ATOL)


def test_SQw_incoh(SQw_incoh_ref, SQw_incoh_obs):

    """
    Validate the calculation of the dynamic incoherent structure factor against
    nMOLDYN

    nMOLDYN normalises all FQt to 1, rather than the incoherent scattering cross
    section, so this factor is included.
    """

    assert np.all(np.shape(SQw_incoh_obs.SQw) == np.shape(SQw_incoh_ref))
    # SQw is normalised to B_FACTOR / N_Q_VALUES.  The first term is due to
    # nMOLDYN not including the incoherent weighting, and the second term is
    # because MDMC normalises the FFT so that there is the same power in SQw as
    # in FQt
    assert_allclose(SQw_incoh_obs.SQw / (B_FACTOR / N_Q_VALUES),
                    SQw_incoh_ref, atol=ATOL)


def test_SQw_coh(SQw_coh_ref, SQw_coh_obs):

    """
    Validate the calculation of the dynamic coherent structure factor against
    nMOLDYN
    """

    assert np.all(np.shape(SQw_coh_obs.SQw) == np.shape(SQw_coh_ref))
    assert_allclose(SQw_coh_obs.SQw * N_Q_VALUES, SQw_coh_ref, atol=ATOL)


def test_SQw_total(SQw_incoh_ref, SQw_coh_ref, SQw_obs):

    """
    Validate the calculation of the dynamic total structure factor against the
    sum of the dynamic incoherent and coherent structure factors calculated by
    nMOLDYN
    """

    assert np.all(np.shape(SQw_obs.SQw) == np.shape(SQw_incoh_ref))
    SQw_ref = (SQw_incoh_ref * B_FACTOR + SQw_coh_ref) / N_Q_VALUES
    assert_allclose(SQw_obs.SQw, SQw_ref, atol=ATOL)
