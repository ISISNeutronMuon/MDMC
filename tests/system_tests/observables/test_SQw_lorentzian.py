"""System tests for total, coherent and incoherent SQw calculations from
MD with a Lorentzian resolution

This compares calculated data with an SQw object generated to be
the Lorentzian function, but designed to make MDMC treat it like a file."""

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

# Values are equivalent to those used to generate the test data
DIMENSIONS = (39.4221067, 39.4221067, 39.4221067)
E_RESOLUTION = {'lorentzian': 49.99998257}

