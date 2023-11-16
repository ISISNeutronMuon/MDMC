"""System tests for total, coherent and incoherent FQt calculations from MD"""

from netCDF4 import Dataset
import numpy as np
from numpy.testing import assert_allclose
import pytest
import periodictable

import MDMC.trajectory_analysis.observables.obs_factory as of
from MDMC.trajectory_analysis.observables import fqt

from tests.test_data import data
from tests.system_tests.observables.data_manager import trajectory, Q_vectors

pytestmark = [pytest.mark.mpi, pytest.mark.lammps]

# Values are equivalent to those used by nMOLDYN to generate the test data
DIMENSIONS = (39.4221067, 39.4221067, 39.4221067)

# Absolute tolerance is included to account for rounding differences in nMOLDYN
# and MDMC
ATOL = 1e-7

# Constants for correct normalisation relative to nMOLDYN FQt and FQt
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
def time_ref(incoh_file):
    return np.array(incoh_file.variables['time'][:])

@pytest.fixture(scope="module")
def FQt_incoh_ref(incoh_file):
    return np.array(incoh_file.variables['Fqt-total'][:])

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
    FQt_coh_ref = (FQt_coh_HH_ref * periodictable.elements.symbol('H').neutron.b_c**2 * N_H
                   + FQt_coh_HO_ref * periodictable.elements.symbol('H').neutron.b_c * periodictable.elements.symbol('O').neutron.b_c * N_H_O
                   + FQt_coh_OO_ref * periodictable.elements.symbol('O').neutron.b_c**2 * N_O) / N_TOTAL
    return FQt_coh_ref

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
def FQt_obs(monkeymodule, trajectory, Q_vectors):
    """
    Setup the container for Q, time, w, total FQt and total SQt
    """

    FQt_total = of.ObservableFactory.create_observable('FQt')
    FQt_total.use_FFT = True
    monkeymodule.setattr(periodictable.elements.symbol('O').neutron,"b_c_i",0)
    FQt_total.calculate_from_MD(trajectory,
                                Q_vectors=Q_vectors,
                                dimensions=DIMENSIONS)
    return FQt_total

@pytest.fixture(scope="module")
def FQt_obs_no_FFT(monkeymodule, trajectory, Q_vectors):
    """
    Setup the container for Q, time, w, total FQt and total SQt
    """

    FQt_total = of.ObservableFactory.create_observable('FQt')
    FQt_total.use_FFT = False
    monkeymodule.setattr(periodictable.elements.symbol('O').neutron,"b_c_i",0)
    FQt_total.calculate_from_MD(trajectory,
                                Q_vectors=Q_vectors,
                                dimensions=DIMENSIONS)
    return FQt_total

@pytest.fixture(scope="module")
def FQt_incoh_obs(monkeymodule, trajectory, Q_vectors):

    """
    Setup the container for Q, time, w, incoherent FQt and incoherent SQt

    Only FQt and SQt are used for testing, as Q, time and w are calculated
    using the same base class as FQt_obs
    """

    FQt_incoh = of.ObservableFactory.create_observable('FQt_incoh')
    FQt_incoh.use_FFT = True
    monkeymodule.setattr(periodictable.elements.symbol('O').neutron,"b_c_i",0)
    FQt_incoh.calculate_from_MD(trajectory,
                                Q_vectors=Q_vectors,
                                dimensions=DIMENSIONS)
    return FQt_incoh

@pytest.fixture(scope="module")
def FQt_incoh_obs_no_FFT(monkeymodule, trajectory, Q_vectors):

    """
    Setup the container for Q, time, w, incoherent FQt and incoherent SQt

    Only FQt and SQt are used for testing, as Q, time and w are calculated
    using the same base class as FQt_obs
    """

    FQt_incoh = of.ObservableFactory.create_observable('FQt_incoh')
    FQt_incoh.use_FFT = False
    monkeymodule.setattr(periodictable.elements.symbol('O').neutron,"b_c_i",0)
    FQt_incoh.calculate_from_MD(trajectory,
                                Q_vectors=Q_vectors,
                                dimensions=DIMENSIONS)
    return FQt_incoh

@pytest.fixture(scope="module")
def FQt_coh_obs(trajectory, Q_vectors):

    """
    Setup the container for Q, time, w, coherent FQt and coherent SQt

    Only FQt and SQt are used for testing, as Q, time and w are calculated
    using the same base class as FQt_obs
    """

    FQt_coh = of.ObservableFactory.create_observable('FQt_coh')
    FQt_coh.use_FFT = True
    FQt_coh.calculate_from_MD(trajectory,
                              Q_vectors=Q_vectors,
                              dimensions=DIMENSIONS)
    return FQt_coh

@pytest.fixture(scope="module")
def FQt_coh_obs_no_FFT(trajectory, Q_vectors):

    """
    Setup the container for Q, time, w, coherent FQt and coherent SQt

    Only FQt and SQt are used for testing, as Q, time and w are calculated
    using the same base class as FQt_obs
    """

    FQt_coh = of.ObservableFactory.create_observable('FQt_coh')
    FQt_coh.use_FFT = False
    FQt_coh.calculate_from_MD(trajectory,
                              Q_vectors=Q_vectors,
                              dimensions=DIMENSIONS)
    return FQt_coh


def test_time(time_ref, FQt_obs):

    """
    Test time equivalence
    """

    # Time in MDMC is in fs, in nMOLDYN is in ps, so factor of 1000 converts
    assert np.all(FQt_obs.t / 1000. == time_ref)

def test_FQt_incoh(FQt_incoh_ref, FQt_incoh_obs):

    """
    Validate the calculation of the intermediate incoherent structure factor
    against nMOLDYN

    nMOLDYN normalises all FQt to 1, rather than the incoherent scattering cross
    section, so this factor is included.
    """

    assert np.all(np.shape(FQt_incoh_obs.FQt) == np.shape(FQt_incoh_ref))
    assert_allclose(FQt_incoh_obs.FQt / B_FACTOR, FQt_incoh_ref, atol=ATOL)

def test_FQt_coh(FQt_coh_ref, FQt_coh_obs):

    """
    Validate the calculation of the intermediate coherent structure factor
    against nMOLDYN
    """

    assert np.all(np.shape(FQt_coh_obs.FQt) == np.shape(FQt_coh_ref))
    assert_allclose(FQt_coh_obs.FQt, FQt_coh_ref, atol=ATOL)

def test_FQt_total(FQt_incoh_ref, FQt_coh_ref, FQt_obs):

    """
    Validate the calculation of the intermediate total structure factor against
    the sum of the intermediate incoherent and coherent structure factors
    calculated by MOLDYN
    """

    assert np.all(np.shape(FQt_obs.FQt) == np.shape(FQt_incoh_ref))
    # Coherent reference is already normalised - do the same for incoherent
    FQt_ref = FQt_incoh_ref * B_FACTOR + FQt_coh_ref
    assert_allclose(FQt_obs.FQt, FQt_ref, atol=ATOL)
