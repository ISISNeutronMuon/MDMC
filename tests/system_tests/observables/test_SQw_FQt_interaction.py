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


def test_fourier_transforms(SQw_obs):
    """
    Tests whether FQt.calculate_SQw and SQw.calculate_FQt are inverse to each other.
    """

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
        resolution = res_type(84.0)

    FQt = SQw_obs.calculate_FQt()
    FQt.apply_resolution(resolution)
    SQw_multiplied = FQt.calculate_SQw()

    SQw_obs.apply_resolution(resolution)

    assert_allclose(SQw_multiplied.SQw, SQw_obs.SQw, atol=1e-07)


