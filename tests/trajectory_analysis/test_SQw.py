"""Tests for SQw observable

Includes calculation from MD trajectory and reading from experimental data file.
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

import MDMC.trajectory_analysis.observables.obs_factory as of
from MDMC.trajectory_analysis.observables.sqw import SQw
import MDMC.trajectory_analysis.trajectory as trj

from tests.test_data import data
from tests.trajectory_analysis.test_histogram import trajectory
from tests.MD.test_simulation import water_SPCE_universe, water_molecule, \
    atom, universe

@pytest.fixture
def altered_trajectory(water_SPCE_universe):

    """
    A list of identical configurations with different times is produced. This
    is passed to Trajectory.
    """

    configurations = []
    times = np.arange(0., 10., 1.)
    for time in times:
        configurations.append(trj.TemporalConfiguration(
            time, *water_SPCE_universe.configuration.atom_list))
    return trj.Trajectory(*configurations)

@pytest.fixture
def SQw_from_data():
    SQw = of.ObservableFactory.create_observable('SQw')
    SQw.read_from_file(reader='LAMPSQw', file_name=data.READER_DATA['LAMPSQw'])
    return SQw

@pytest.fixture
def SQw_from_MD(trajectory, universe) -> callable:

    """
    Returns
    -------
    callable
        A function which optionally accepts ``use_FFT`` (defaults to `True`) and ``use_traj_list``
        (defaults to `False`). Returns an ``SQw`` ``Observable``.
    """

    def _SQw_from_MD(use_FFT: bool = True, use_traj_list: bool = False) -> SQw:
        _SQw = of.ObservableFactory.create_observable('SQw')
        _SQw.use_FFT = use_FFT
        dimensions = universe.dimensions
        n_Q = 10
        energy_resolution = 49.99998257
        Q_values = [2 * np.pi * i / dimensions[0] for i in range(1, n_Q+1)]

        if use_traj_list:
            MD_input = [trajectory]
        else:
            MD_input = trajectory

        _SQw.calculate_from_MD(MD_input,
                               Q_values=Q_values,
                               dimensions=dimensions,
                               energy_resolution=energy_resolution)
        return _SQw

    return _SQw_from_MD

# TODO: Test for consistency by comparing S(Q,w) where w = 0 with S(Q)


def test_from_data(SQw_from_data):

    """
    Test the following:

    - _from_MD flag is False
    - reader is LAMPSQw
    - Q and E are the independent variables
    - SQw is the dependent variable
    - SQw is the variable on which there is an error
    - Q ranges from 0 to 3.5 in 0.05 increments
    """

    assert SQw_from_data.origin == 'experiment'
    assert SQw_from_data.reader.__class__.__name__ == "LAMPSQw"

    assert 'Q' in SQw_from_data.independent_variables and \
        'E' in SQw_from_data.independent_variables
    assert 'SQw' in SQw_from_data.dependent_variables
    assert 'SQw' in SQw_from_data.errors

    # Cannot use assert_allclose as our UnitNDArray fails comparison with a
    # normal numpy array
    for i, Q in enumerate(SQw_from_data.independent_variables['Q']):
        assert np.isclose(Q, i * 0.05)


def test_from_MD(SQw_from_MD):

    """
    Test the following:
    - ``origin`` is 'MD'
    - reader is LAMPSQw
    - Q and E are the independent variables
    - SQw is the dependent variable
    - SQw is the variable on which there is an error
    - Q ranges from 0 to 3.5 in 0.05 increments
    - SQw is the same whether FFT is used or not
    - SQw handles either a single, or ``list`` of, ``Trajectory`` objects
    """

    SQw_FFT = SQw_from_MD()
    SQw_no_FFT = SQw_from_MD(use_FFT=False, use_traj_list=True)

    assert SQw_FFT.origin == 'MD'
    assert 'Q' in SQw_FFT.independent_variables and \
        'E' in SQw_FFT.independent_variables
    assert 'SQw' in SQw_FFT.dependent_variables
    assert 'SQw' in SQw_FFT.errors

    # Recreate the momentum values we create in SQw_from_MD to assert against
    assert_allclose(SQw_FFT.independent_variables['Q'],
                    2 * np.pi * np.arange(0.1, 1.1, 0.1))

    # Assert there is no difference between FFT and non-FFT calculation
    assert_allclose(SQw_FFT.SQw, SQw_no_FFT.SQw, rtol=1e-5)


def test_trajectory_assertions(SQw_from_MD, trajectory, altered_trajectory):

    """
    Test that an ``AssertionError`` is raised when a list of trajectories that have different times
    are given to ``calculate_from_MD()``
    """

    MD_input = [trajectory, altered_trajectory]
    SQw_obj = SQw_from_MD()
    with pytest.raises(AssertionError):
        SQw_obj.calculate_from_MD(MD_input, energy_resolution=1.)
