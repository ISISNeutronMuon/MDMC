import pytest
from numpy.testing import assert_allclose

from MDMC.common.unit_registry import *


@pytest.mark.parametrize('MDMC_unit, LAMMPS_unit', [(1 * UREG.angstrom, 1 * UREG.angstrom),
                                                    (1 * UREG.fs, 1 * UREG.fs),
                                                    (1 * UREG.amu, 1 * UREG.g / UREG.mol),
                                                    (1 * UREG.e, 1 * UREG.e),
                                                    (1 * UREG.deg, 1 * UREG.deg),
                                                    (1 * UREG.kelvin, 1 * UREG.kelvin),
                                                    (1 * UREG.kJ / UREG.mol, 0.239005736 * UREG.kcal / UREG.mol),
                                                    (1 * UREG.kJ / (UREG.angstrom * UREG.mol), 0.239005736 * UREG.kcal / (UREG.angstrom * UREG.mol)),
                                                    (1 * UREG.Pa, 9.86923267e-06 * UREG.atm)])
def test_convert_unit(MDMC_unit, LAMMPS_unit):
    """
    Test that unit conversion works as intended.
    Uses allclose instead of equality to account for rounding errors.
    """

    assert_allclose(convert_unit(MDMC_unit, target='LAMMPS'), LAMMPS_unit, atol=1e-05)
    assert_allclose(convert_unit(LAMMPS_unit, target='MDMC'), MDMC_unit, atol=1e-05)


@pytest.mark.parametrize('obj', [(5 * UREG.cm), 8, 'string', True, None])
def test_strip_unit(obj):
    stripped_obj = strip_unit(obj)
    if isinstance(obj, pint.Quantity):
        assert stripped_obj == 5
    else:
        assert stripped_obj == obj
