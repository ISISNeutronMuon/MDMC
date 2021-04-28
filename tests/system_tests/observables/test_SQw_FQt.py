"""System tests for total, coherent and incoherent SQw and FQt calculations from
MD

Although SQw and FQt are two separate observables, as the calculation of SQw
realies on the calculation of FQt they are tested together."""

from netCDF4 import Dataset
import numpy as np
from numpy.testing import assert_allclose
import pytest

import MDMC.common.atom_properties as ap
import MDMC.trajectory_analysis.observables.obs_factory as of
from MDMC.trajectory_analysis.observables import sqw
from MDMC.trajectory_analysis.observables.sqw_coh import SQwCoherent
from MDMC.trajectory_analysis.observables.sqw_incoh import SQwIncoherent


from tests.test_data import data
from tests.system_tests.observables.data_manager import trajectory, Q_vectors

pytestmark = pytest.mark.mpi

# Values are equivalent to those used by nMOLDYN to generate the test data
DIMENSIONS = (39.4221067, 39.4221067, 39.4221067)
E_RESOLUTION = 49.99998257

# Absolute tolerance is included to account for rounding differences in nMOLDYN
# and MDMC
ATOL = 1e-7

# Constants for correct normalisation relative to nMOLDYN FQt and SQw
N_H = 4096
N_O = 2048
N_TOTAL = N_H + N_O
N_H_O = np.sqrt(N_H * N_O)
# B_FACTOR is set to a constant rather than calculated using B_COH and B_INCOH
# as MDMC has a different oxygen B_INCOH value to nMOLDYN
# B_FACTOR = (ap.B_INCOH['H']**2 * N_H + ap.B_INCOH['O']**2 * N_O) / N_TOTAL
B_FACTOR = 425.792524267738
N_Q_VALUES = 13

@pytest.fixture(scope="module")
def incoh_file():
    return Dataset(data.OBS_DATA['SQw_incoh'], 'r')

@pytest.fixture(scope="module")
def coh_file():
    return Dataset(data.OBS_DATA['SQw_coh'], 'r')

@pytest.fixture(scope="module")
def Q_ref(incoh_file):
    return np.array(incoh_file.variables['q'][:])

@pytest.fixture(scope="module")
def time_ref(incoh_file):
    return np.array(incoh_file.variables['time'][:])

@pytest.fixture(scope="module")
def w_ref(incoh_file):
    # nMOLDYN test file has 50 points in time and frequency, however we can
    # only generate 49 energy points from 50 frames so crop and rescale array
    w_raw = np.array(incoh_file.variables['angular_frequency'][:])
    return w_raw[:-1] * len(w_raw) / (len(w_raw) - 1)

@pytest.fixture(scope="module")
def FQt_incoh_ref(incoh_file):
    return np.array(incoh_file.variables['Fqt-total'][:])

@pytest.fixture(scope="module")
def SQw_incoh_ref(incoh_file):
    # nMOLDYN test file has 50 points in time and frequency, however we can
    # only generate 49 energy points from 50 frames so crop the array in energy
    return np.array(incoh_file.variables['Sqw-total'][:])[:, :-1]

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
    # nMOLDYN test file has 50 points in time and frequency, however we can
    # only generate 49 energy points from 50 frames so crop the array in energy
    SQw_coh_ref = (SQw_coh_HH_ref * ap.B_COH['H']**2 * N_H
                   + SQw_coh_HO_ref * ap.B_COH['H'] * ap.B_COH['O'] * N_H_O
                   + SQw_coh_OO_ref * ap.B_COH['O']**2 * N_O) / N_TOTAL
    return SQw_coh_ref[:, :-1]

@pytest.fixture(scope='module')
def monkeymodule():

    """
    This is an ugly workaround because pytest does not currently allow
    monkeypatch to be used in module scoped fixtures.
    """

    from _pytest.monkeypatch import MonkeyPatch
    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()

@pytest.fixture(scope="module")
def SQw_obs(monkeymodule, trajectory, Q_vectors):

    """
    Returns
    -------
    callable
        A function which optionally accepts ``use_FFT`` (defaults to `True`)
        and returns an ``SQw`` ``Observable``.
    """

    def _SQw_obs(use_FFT: bool = True) -> sqw.SQw:

        """
        Setup the container for Q, time, w, total FQt and total SQt
        """

        SQw_total = of.ObservableFactory.create_observable('SQw')
        SQw_total.use_FFT = use_FFT
        monkeymodule.setitem(sqw.B_INCOH, 'O', 0.)
        SQw_total.calculate_from_MD(trajectory,
                                    Q_vectors=Q_vectors,
                                    dimensions=DIMENSIONS,
                                    energy_resolution=E_RESOLUTION)
        return SQw_total

    return _SQw_obs

@pytest.fixture(scope="module")
def SQw_incoh_obs(monkeymodule, trajectory, Q_vectors):

    """
    Returns
    -------
    callable
        A function which optionally accepts ``use_FFT`` (defaults to `True`)
        and returns an ``SQw`` ``Observable``.
    """

    def _SQw_obs(use_FFT: bool = True) -> SQwIncoherent:

        """
        Setup the container for Q, time, w, incoherent FQt and incoherent SQt

        Only FQt and SQt are used for testing, as Q, time and w are calculated
        using the same base class as SQw_obs
        """

        SQw_incoh = of.ObservableFactory.create_observable('SQw_incoh')
        SQw_incoh.use_FFT = use_FFT
        monkeymodule.setitem(sqw.B_INCOH, 'O', 0.)
        SQw_incoh.calculate_from_MD(trajectory,
                                    Q_vectors=Q_vectors,
                                    dimensions=DIMENSIONS,
                                    energy_resolution=E_RESOLUTION)
        return SQw_incoh

    return _SQw_obs

@pytest.fixture(scope="module")
def SQw_coh_obs(trajectory, Q_vectors):

    """
    Returns
    -------
    callable
        A function which optionally accepts ``use_FFT`` (defaults to `True`)
        and returns an ``SQw`` ``Observable``.
    """

    def _SQw_obs(use_FFT: bool = True) -> SQwCoherent:

        """
        Setup the container for Q, time, w, coherent FQt and coherent SQt

        Only FQt and SQt are used for testing, as Q, time and w are calculated
        using the same base class as SQw_obs
        """

        SQw_coh = of.ObservableFactory.create_observable('SQw_coh')
        SQw_coh.use_FFT = use_FFT
        SQw_coh.calculate_from_MD(trajectory,
                                  Q_vectors=Q_vectors,
                                  dimensions=DIMENSIONS,
                                  energy_resolution=E_RESOLUTION)
        return SQw_coh

    return _SQw_obs


def test_time(time_ref, SQw_obs):

    """
    Test time equivalence
    """

    # Time in MDMC is in fs, in nMOLDYN is in ps, so factor of 1000 converts
    assert np.all(SQw_obs().t / 1000. == time_ref)


def test_w(w_ref, SQw_obs):

    """
    Test angular frequency equivalence

    Exact equivalence results in failed assertion due to rounding errors
    """

    assert_allclose(SQw_obs().w, w_ref, atol=1e-07)


def test_FQt_incoh(FQt_incoh_ref, SQw_incoh_obs):

    """
    Validate the calculation of the intermediate incoherent structure factor
    against nMOLDYN

    nMOLDYN normalises all FQt to 1, rather than the incoherent scattering cross
    section, so this factor is included.
    """

    assert np.all(np.shape(SQw_incoh_obs().FQt) == np.shape(FQt_incoh_ref))
    assert_allclose(SQw_incoh_obs().FQt / B_FACTOR, FQt_incoh_ref, atol=ATOL)


def test_FQt_coh(FQt_coh_ref, SQw_coh_obs):

    """
    Validate the calculation of the intermediate coherent structure factor
    against nMOLDYN
    """

    assert np.all(np.shape(SQw_coh_obs().FQt) == np.shape(FQt_coh_ref))
    assert_allclose(SQw_coh_obs().FQt, FQt_coh_ref, atol=ATOL)


def test_FQt_total(FQt_incoh_ref, FQt_coh_ref, SQw_obs):

    """
    Validate the calculation of the intermediate total structure factor against
    the sum of the intermediate incoherent and coherent structure factors
    calculated by MOLDYN
    """

    assert np.all(np.shape(SQw_obs().FQt) == np.shape(FQt_incoh_ref))
    # Coherent reference is already normalised - do the same for incoherent
    FQt_ref = FQt_incoh_ref * B_FACTOR + FQt_coh_ref
    assert_allclose(SQw_obs().FQt, FQt_ref, atol=ATOL)


def test_SQw_incoh(SQw_incoh_ref, SQw_incoh_obs):

    """
    Validate the calculation of the dynamic incoherent structure factor against
    nMOLDYN

    nMOLDYN normalises all FQt to 1, rather than the incoherent scattering cross
    section, so this factor is included.
    """

    assert np.all(np.shape(SQw_incoh_obs().SQw) == np.shape(SQw_incoh_ref))
    # SQw is normalised to B_FACTOR / N_Q_VALUES.  The first term is due to
    # nMOLDYN not including the incoherent weighting, and the second term is
    # because MDMC normalises the FFT so that there is the same power in SQw as
    # in FQt
    assert_allclose(SQw_incoh_obs().SQw / (B_FACTOR / N_Q_VALUES),
                    SQw_incoh_ref, atol=ATOL)

    # Assert there is no difference between FFT and non-FFT calculation
    assert_allclose(SQw_incoh_obs().SQw, SQw_incoh_obs(use_FFT=False).SQw, atol=ATOL)


def test_SQw_coh(SQw_coh_ref, SQw_coh_obs):

    """
    Validate the calculation of the dynamic coherent structure factor against
    nMOLDYN
    """

    assert np.all(np.shape(SQw_coh_obs().SQw) == np.shape(SQw_coh_ref))
    assert_allclose(SQw_coh_obs().SQw * N_Q_VALUES, SQw_coh_ref, atol=ATOL)

    # Assert there is no difference between FFT and non-FFT calculation
    assert_allclose(SQw_coh_obs().SQw, SQw_coh_obs(use_FFT=False).SQw, atol=ATOL)


def test_SQw_total(SQw_incoh_ref, SQw_coh_ref, SQw_obs):

    """
    Validate the calculation of the dynamic total structure factor against the
    sum of the dynamic incoherent and coherent structure factors calculated by
    nMOLDYN
    """

    assert np.all(np.shape(SQw_obs().SQw) == np.shape(SQw_incoh_ref))
    SQw_ref = (SQw_incoh_ref * B_FACTOR + SQw_coh_ref) / N_Q_VALUES
    assert_allclose(SQw_obs().SQw, SQw_ref, atol=ATOL)

    # Assert there is no difference between FFT and non-FFT calculation
    assert_allclose(SQw_obs().SQw, SQw_obs(use_FFT=False).SQw, atol=ATOL)
