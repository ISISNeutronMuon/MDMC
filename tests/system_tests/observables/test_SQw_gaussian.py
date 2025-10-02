"""System tests for total, coherent and incoherent SQw and FQt calculations from
MD with a Gaussian resolution

Although SQw and FQt are two separate observables, as the calculation of SQw
relies on the calculation of FQt they are tested together."""

from netCDF4 import Dataset
import numpy as np
from numpy.testing import assert_allclose
import pytest
import periodictable

import MDMC.trajectory_analysis.observables.obs_factory as of

from tests.test_data import data
from tests.system_tests.observables.data_manager import trajectory, Q_vectors

pytestmark = [pytest.mark.mpi, pytest.mark.lammps]

# Values are equivalent to those used by nMOLDYN to generate the test data
DIMENSIONS = (39.4221067, 39.4221067, 39.4221067)
E_RESOLUTION = {'gaussian': 49.99998257}

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
# calc_incoherent_scatt_length needs to be imported from MDMC.trajectory_analysis.observables.fqt.py
# B_FACTOR = (calc_incoherent_scatt_length('H')**2 * N_H + calc_incoherent_scatt_length('O')**2 * N_O) / N_TOTAL
B_FACTOR = 425.7925244185173
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
def w_ref(incoh_file):
    # nMOLDYN test file has 50 points in time and frequency, however we can
    # only generate 49 energy points from 50 frames so crop and rescale array
    w_raw = np.array(incoh_file.variables['angular_frequency'][:])
    return w_raw[:-1] * len(w_raw) / (len(w_raw) - 1)

@pytest.fixture(scope="module")
def SQw_incoh_ref(incoh_file):
    # nMOLDYN test file has 50 points in time and frequency, however we can
    # only generate 49 energy points from 50 frames so crop the array in energy
    return np.array(incoh_file.variables['Sqw-total'][:])[:, :-1]

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
    SQw_coh_ref = (SQw_coh_HH_ref * periodictable.elements.symbol('H').neutron.b_c**2 * N_H
                   + SQw_coh_HO_ref * periodictable.elements.symbol('H').neutron.b_c* periodictable.elements.symbol('O').neutron.b_c * N_H_O
                   + SQw_coh_OO_ref * periodictable.elements.symbol('O').neutron.b_c**2 * N_O) / N_TOTAL
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
    Setup the container for Q, time, w, total FQt and total SQt
    """

    SQw_total = of.ObservableFactory.create('SQw')
    SQw_total.use_FFT = True
    monkeymodule.setattr(periodictable.elements.symbol('O').neutron,"b_c_i",0)
    SQw_total.calculate_from_MD(trajectory,
                                Q_vectors=Q_vectors,
                                dimensions=DIMENSIONS,
                                energy_resolution=E_RESOLUTION)
    return SQw_total

@pytest.fixture(scope="module")
def SQw_obs_no_FFT(monkeymodule, trajectory, Q_vectors):
    """
    Setup the container for Q, time, w, total FQt and total SQt
    """

    SQw_total = of.ObservableFactory.create('SQw')
    SQw_total.use_FFT = False
    monkeymodule.setattr(periodictable.elements.symbol('O').neutron,"b_c_i",0)
    SQw_total.calculate_from_MD(trajectory,
                                Q_vectors=Q_vectors,
                                dimensions=DIMENSIONS,
                                energy_resolution=E_RESOLUTION)
    return SQw_total

@pytest.fixture(scope="module")
def SQw_incoh_obs(monkeymodule, trajectory, Q_vectors):

    """
    Setup the container for Q, time, w, incoherent FQt and incoherent SQt

    Only FQt and SQt are used for testing, as Q, time and w are calculated
    using the same base class as SQw_obs
    """

    SQw_incoh = of.ObservableFactory.create('SQw_incoh')
    SQw_incoh.use_FFT = True
    monkeymodule.setattr(periodictable.elements.symbol('O').neutron,"b_c_i",0)
    SQw_incoh.calculate_from_MD(trajectory,
                                Q_vectors=Q_vectors,
                                dimensions=DIMENSIONS,
                                energy_resolution=E_RESOLUTION)
    return SQw_incoh

@pytest.fixture(scope="module")
def SQw_incoh_obs_no_FFT(monkeymodule, trajectory, Q_vectors):

    """
    Setup the container for Q, time, w, incoherent FQt and incoherent SQt

    Only FQt and SQt are used for testing, as Q, time and w are calculated
    using the same base class as SQw_obs
    """

    SQw_incoh = of.ObservableFactory.create('SQw_incoh')
    SQw_incoh.use_FFT = False
    monkeymodule.setattr(periodictable.elements.symbol('O').neutron,"b_c_i",0)
    SQw_incoh.calculate_from_MD(trajectory,
                                Q_vectors=Q_vectors,
                                dimensions=DIMENSIONS,
                                energy_resolution=E_RESOLUTION)
    return SQw_incoh

@pytest.fixture(scope="module")
def SQw_coh_obs(trajectory, Q_vectors):

    """
    Setup the container for Q, time, w, coherent FQt and coherent SQt

    Only FQt and SQt are used for testing, as Q, time and w are calculated
    using the same base class as SQw_obs
    """

    SQw_coh = of.ObservableFactory.create('SQw_coh')
    SQw_coh.use_FFT = True
    SQw_coh.calculate_from_MD(trajectory,
                              Q_vectors=Q_vectors,
                              dimensions=DIMENSIONS,
                              energy_resolution=E_RESOLUTION)
    return SQw_coh

@pytest.fixture(scope="module")
def SQw_coh_obs_no_FFT(trajectory, Q_vectors):

    """
    Setup the container for Q, time, w, coherent FQt and coherent SQt

    Only FQt and SQt are used for testing, as Q, time and w are calculated
    using the same base class as SQw_obs
    """

    SQw_coh = of.ObservableFactory.create('SQw_coh')
    SQw_coh.use_FFT = False
    SQw_coh.calculate_from_MD(trajectory,
                              Q_vectors=Q_vectors,
                              dimensions=DIMENSIONS,
                              energy_resolution=E_RESOLUTION)
    return SQw_coh


def test_w(w_ref, SQw_obs):

    """
    Test angular frequency equivalence

    Exact equivalence results in failed assertion due to rounding errors
    """

    assert_allclose(SQw_obs.w, w_ref, atol=1e-07)


def test_SQw_incoh(SQw_incoh_ref, SQw_incoh_obs, SQw_incoh_obs_no_FFT):

    """
    Validate the calculation of the dynamic incoherent structure factor against
    nMOLDYN

    nMOLDYN normalises all FQt to 1, rather than the incoherent scattering cross
    section, so this factor is included.
    """

    assert np.all(np.shape(SQw_incoh_obs.SQw[0]) == np.shape(SQw_incoh_ref))
    # SQw is normalised to B_FACTOR / N_Q_VALUES.  The first term is due to
    # nMOLDYN not including the incoherent weighting, and the second term is
    # because MDMC normalises the FFT so that there is the same power in SQw as
    # in FQt
    assert_allclose(SQw_incoh_obs.SQw[0] / (B_FACTOR / N_Q_VALUES),
                    SQw_incoh_ref, atol=ATOL)

    # Assert there is no difference between FFT and non-FFT calculation
    assert_allclose(SQw_incoh_obs.SQw[0], SQw_incoh_obs_no_FFT.SQw[0], atol=ATOL)


def test_SQw_coh(SQw_coh_ref, SQw_coh_obs, SQw_coh_obs_no_FFT):

    """
    Validate the calculation of the dynamic coherent structure factor against
    nMOLDYN
    """

    assert np.all(np.shape(SQw_coh_obs.SQw[0]) == np.shape(SQw_coh_ref))
    assert_allclose(SQw_coh_obs.SQw[0] * N_Q_VALUES, SQw_coh_ref, atol=ATOL)

    # Assert there is no difference between FFT and non-FFT calculation
    assert_allclose(SQw_coh_obs.SQw[0], SQw_coh_obs_no_FFT.SQw[0], atol=ATOL)


def test_SQw_total(SQw_incoh_ref, SQw_coh_ref, SQw_obs, SQw_obs_no_FFT):

    """
    Validate the calculation of the dynamic total structure factor against the
    sum of the dynamic incoherent and coherent structure factors calculated by
    nMOLDYN
    """

    assert np.all(np.shape(SQw_obs.SQw[0]) == np.shape(SQw_incoh_ref))
    SQw_ref = (SQw_incoh_ref * B_FACTOR + SQw_coh_ref) / N_Q_VALUES
    assert_allclose(SQw_obs.SQw[0], SQw_ref, atol=ATOL)

    # Assert there is no difference between FFT and non-FFT calculation
    assert_allclose(SQw_obs.SQw[0], SQw_obs_no_FFT.SQw[0], atol=ATOL)
