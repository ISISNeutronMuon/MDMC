"""System tests for total, coherent and incoherent SQw calculations with
maximum times shorter than the provided trajectories

The maximum time, t, for a required trajectory for calculating SQw depends on
the SQw energy step size, dE.  If the trajectory provided has a larger t than is
required by dE, SQw must still be calculated for dE step sizes.  These unit
tests ensure that SQw is the same (within uncertainty) independent of the
trajectory length, it the same energies are specified.  THIS MODULE COULD BE
PARAMETERIZED TO TEST OTHER OBSERVABLES"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from MDMC.common.constants import h
from MDMC.trajectory_analysis.observables.obs_factory import ObservableFactory
from tests.system_tests.observables.data_manager import trajectory

pytestmark = [pytest.mark.mpi, pytest.mark.lammps]


ATOL = 1e-7


@pytest.fixture(scope="module")
def independent_variables(trajectory):

    """
    Calculate the independent variables

    E is equivalent to the times from half the trajectory length

    Returns:
    Dictionary of independent variables required for SQw, SQw_coh, and SQw_incoh
    """

    # Use half the trajectory steps to calculate the Energies, taking into
    # account the fact we get n energy points from n + 1 frames
    n = int(len(trajectory.times) / 2 - 1)
    dt = trajectory.times[1] - trajectory.times[0]
    # h is in units of eV s whereas system units are meV fs, so apply a
    # factor of 1e3 * 1e15 to convert it
    E = h * 1e18 * np.fft.fftfreq(2 * n, dt)[:n]
    Q = np.arange(1.6, 21, 1.6)

    return {'E':E, 'Q':Q}

@pytest.fixture(params=['SQw', 'SQw_coh', 'SQw_incoh'])
def SQw_type(request):

    """
    SQw_type is parameterized with the strings required to create SQw, SQw_coh
    and SQw_incoh observable types
    """

    return request.param


def test_SQw_max_t(trajectory, independent_variables, SQw_type):

    """
    Tests the total SQw with times shorter than provided the trajectory

    Three SQw are calculated, one using the full trajectory, one using the first
    half of the trajectory, and one using the second half of the trajectory.
    All SQw are calculated for the same values of Q and E.  The SQw calculated
    from the total trajectory is tested for consistency with the two half
    trajectory SQws.
    """

    E_RES = 49.99998257
    DIMENSIONS = [39.42210674, 39.42210674, 39.42210674]

    SQw_observable = ObservableFactory.create(SQw_type)
    SQw_observable.independent_variables = independent_variables
    n = len(trajectory.times) // 2

    SQw_observable.calculate_from_MD(trajectory,
                                     energy_resolution=E_RES,
                                     dimensions=DIMENSIONS)
    SQw_full_array = SQw_observable.SQw[0]

    SQw_observable.calculate_from_MD(trajectory[:n],
                                     energy_resolution=E_RES,
                                     dimensions=DIMENSIONS)
    SQw_1_array = SQw_observable.SQw[0]

    SQw_observable.calculate_from_MD(trajectory[n:],
                                     energy_resolution=E_RES,
                                     dimensions=DIMENSIONS)
    SQw_2_array = SQw_observable.SQw[0]

    # Calculate the total standard deviation for the two half runs and test that
    # the total run is within a factor of 3
    SQw_1_2_mean = np.mean([SQw_1_array, SQw_2_array], axis=0)
    stdev = np.std([SQw_1_array, SQw_2_array], axis=0)
    stdev_total = np.sum(stdev)
    stdev_full = np.std([SQw_1_2_mean, SQw_full_array], axis=0)
    assert np.sum(stdev_full) < 3 * stdev_total

    # Test that the stdev for each Q,w value for the total run is within a
    # factor of 2 of the maximum standard deviation of any point
    assert np.all(stdev_full < 2 * np.max(stdev))

    # Test without FFT
    SQw_observable.use_FFT = False

    SQw_observable.calculate_from_MD(trajectory,
                                     energy_resolution=E_RES,
                                     dimensions=DIMENSIONS)
    SQw_full_array_no_FFT = SQw_observable.SQw[0]

    SQw_observable.calculate_from_MD(trajectory[:n],
                                     energy_resolution=E_RES,
                                     dimensions=DIMENSIONS)
    SQw_1_array_no_FFT = SQw_observable.SQw[0]

    SQw_observable.calculate_from_MD(trajectory[n:],
                                     energy_resolution=E_RES,
                                     dimensions=DIMENSIONS)
    SQw_2_array_no_FFT = SQw_observable.SQw[0]

    # Assert there is no difference between FFT and non-FFT calculation for
    # the short trajectories, as in both cases all frames are utilised
    assert_allclose(SQw_1_array, SQw_1_array_no_FFT, atol=ATOL)
    assert_allclose(SQw_2_array, SQw_2_array_no_FFT, atol=ATOL)

    # For the full trajectory, the FFT method will only utilise the first n
    # frames. The non-FFT method uses all the frames, and so we cannot assert
    # that the two are "allclose" in general. Instead, apply the same criteria
    # of being within a number of standard deviations as we did before
    stdev_full_no_FFT = np.std([SQw_1_2_mean, SQw_full_array_no_FFT], axis=0)
    assert np.sum(stdev_full_no_FFT) < 3 * stdev_total
    assert np.all(stdev_full_no_FFT < 2 * np.max(stdev))
