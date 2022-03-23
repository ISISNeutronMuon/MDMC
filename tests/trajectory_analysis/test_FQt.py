"""Tests for FQt observable

Tests various parts of FQt MD calculation
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from typing import Optional

import MDMC.trajectory_analysis.observables.obs_factory as of
from MDMC.trajectory_analysis.observables.fqt import FQt

from tests.trajectory_analysis.test_histogram import trajectory
from tests.MD.test_simulation import water_SPCE_universe, water_molecule, \
    atom, universe
from tests.test_data.calculated_values.Q_vectors import Q_VECTORS

@pytest.fixture
def FQt_from_MD(trajectory, universe):
    def _FQt_from_MD() -> FQt:
        _FQt = of.ObservableFactory.create_observable('FQt')
        dimensions = universe.dimensions
        n_Q = 10
        Q_values = [2 * np.pi * i / dimensions[0] for i in range(1, n_Q + 1)]

        _FQt.calculate_from_MD(trajectory,
                               Q_values=Q_values,
                               dimensions=dimensions)

        return _FQt

    return _FQt_from_MD


def test_calculate_Q_vectors(FQt_from_MD):
    """Tests that Q-vectors are calculated correctly."""
    FQt = FQt_from_MD()

    for i in range(len(Q_VECTORS)):
        assert_allclose(FQt.Q_vectors[i], Q_VECTORS[i])
