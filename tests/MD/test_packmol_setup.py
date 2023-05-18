from MDMC.MD.packmol.packmol_setup import PackmolSetup
import pytest
import numpy as np
from pytest_cases import fixture

import MDMC.utilities.packmol_wrapper as packmol
import tests.test_data.data as test_data
from MDMC.MD import Universe
from MDMC.MD.packmol.packmol_setup import calculate_volume

# lammps mark used to ensure test runs in docker container
pytestmark = [pytest.mark.lammps]

@pytest.mark.parametrize('lengths, container_type, expected',
                         [(20., "cube", 800.),
                          (10., "sphere", 4188.8),
                          ([10., 20., 5.], "box", 1000.)])
def test_get_volume_affecting_dimensions(lengths, container_type, expected):
    volume = calculate_volume(lengths, container_type)
    assert np.close(volume, expected)

