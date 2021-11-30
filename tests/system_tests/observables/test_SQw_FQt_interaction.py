"""
Tests for interactions between SQw and FQt objects.
"""
from numpy.testing import assert_allclose
import pytest

import MDMC.trajectory_analysis.observables.obs_factory as of
import MDMC.resolution as res
from tests.test_data import data

# get resolution types for test_apply_resolution
resdict = res.ResolutionFactory().resolutions
resparms = list(resdict.values())

@pytest.fixture
def SQw_obs():
    """
    An SQw observable for test use.
    """
    SQw = of.ObservableFactory.create_observable('SQw')
    SQw.read_from_file(reader='LAMPSQw', file_name=data.READER_DATA['LAMPSQw'])
    return SQw


@pytest.mark.parametrize('use_FFT', [True, False])
def test_fourier_transforms(SQw_obs, use_FFT):
    """
    Tests whether FQt.calculate_SQw and SQw.calculate_FQt are inverse to each other.
    """
    SQw_obs.use_FFT = use_FFT

    FQt = SQw_obs.calculate_FQt()
    SQw_transformed = FQt.calculate_SQw()

    assert SQw_transformed.Q.all() == SQw_obs.Q.all()
    assert SQw_transformed.SQw.all() == SQw_obs.SQw.all()
    assert_allclose(SQw_transformed.E, SQw_obs.E, atol=1e-07)


@pytest.mark.parametrize("res_type", resparms)
def test_apply_resolution(SQw_obs, res_type):
    """
    Tests whether applying resolution via convolution or multiplication give the same result.

    Essentially,tests whether the following diagram commutes:

               R(Q,w)
    SQw_ideal -------> SQw_real
      |                     ^
      | F^-1                | F
      V                     |
    FQt_ideal -------> FQt_real
               R(Q,t)

    """

    if res_type == res.FileResolution:
        resolution = res_type(data.RESOLUTION_DATA['LAMPSQw'],
                              'SQw', 'LAMPSQw', 1055.8303421611213)
    else:
        resolution = res_type(100.0)

    SQw_convolved = SQw_obs

    FQt = SQw_obs.calculate_FQt()
    FQt.apply_resolution(resolution)
    SQw_multiplied = FQt.calculate_SQw()

    SQw_convolved.apply_resolution(resolution)
    # we convolve the array back and forth to sort out any potential format quirks;
    # see test_fourier_transform to see that this doesn't affect the actual array
    FQt_convolved = SQw_convolved.calculate_FQt()
    SQw_convolved = FQt_convolved.calculate_SQw()

    assert_allclose(SQw_multiplied.SQw, SQw_convolved.SQw, atol=1e-07)
