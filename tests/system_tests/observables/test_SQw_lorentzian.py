"""System tests for total, coherent and incoherent SQw calculations from
MD with a Lorentzian resolution"""

from netCDF4 import Dataset
import numpy as np
from numpy.testing import assert_allclose
import pytest

import MDMC.trajectory_analysis.observables.obs_factory as of
from MDMC.trajectory_analysis.observables import sqw

from tests.test_data import data
from tests.system_tests.observables.data_manager import trajectory, Q_vectors

pytestmark = [pytest.mark.lammps]

# TODO
# see https://github.com/MDMCproject/MDMCv0.2_pilot/issues/737
