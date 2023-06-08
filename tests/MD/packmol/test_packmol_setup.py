import pytest
import numpy as np

from MDMC.MD.packmol.packmol_setup import calculate_volume

# lammps mark used to ensure test runs in docker container
pytestmark = [pytest.mark.lammps]

@pytest.mark.parametrize('lengths, container_type, expected',
                         [((20.,), "cube", 8000.),
                          ((10.,), "sphere", 4188.79),
                          ((10., 20., 5.), "box", 1000.)])
def test_calculate_volume(lengths, container_type, expected):
    volume = calculate_volume(lengths, container_type)
    assert np.isclose(volume, expected)

