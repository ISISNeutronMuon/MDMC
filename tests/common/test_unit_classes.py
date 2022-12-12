"""Tests any classes that add units to the representation"""

from copy import deepcopy

import numpy as np
import pytest
from pytest_cases import parametrize, fixture_ref

from MDMC.common.mathematics import correlation
from MDMC.common.units import Unit, UnitFloat, UnitNDArray, unit_array


FLOAT = 50.0
LIST = [FLOAT, FLOAT, FLOAT]
ARRAY = np.array(LIST)
UNIT = 'fs'


@pytest.fixture
def ufloat():

    return UnitFloat(FLOAT, UNIT)


@pytest.fixture
def uarray():

    uarr = UnitNDArray(len(LIST), UNIT)
    uarr[:] = LIST
    return uarr


@pytest.fixture
def uarray_helper():

    return unit_array(LIST, UNIT)


@parametrize("uclass", [fixture_ref(ufloat), fixture_ref(uarray)])
def test_unit_deepcopy(uclass):
    """
    Tests __deepcopy__ for all classes that add units to the representation
    """

    cpy_uclass = deepcopy(uclass)
    for attr in uclass.__dict__:
        assert np.all(getattr(uclass, attr) == getattr(cpy_uclass, attr))

    cpy_uclass.unit = 'mg'
    assert cpy_uclass.unit != uclass.unit


@parametrize("uclass", [fixture_ref(ufloat), fixture_ref(uarray), fixture_ref(uarray_helper)])
def test_unit_is_Unit(uclass):
    """
    Tests that for all classes that add units to the representation, and all
    helper functions that create these classes, the unit must be a Unit object
    """

    assert isinstance(uclass.unit, Unit)


@parametrize("uclass", [fixture_ref(ufloat), fixture_ref(uarray), fixture_ref(uarray_helper)])
def test_repr_contains_unit(uclass):
    """
    Tests that for all classes that add units to the representation, all
    representations contain the units
    """

    assert UNIT in repr(uclass)


@pytest.mark.parametrize("op,args", [(np.exp, ()),
                                     (np.sin, ()),
                                     (np.sum, ()),
                                     (np.mean, ()),
                                     (np.power, (2., )),
                                     (correlation, ())
                                    ])
def test_array_operations(op, args, uarray):
    """
    Tests math operations (e.g. np.ufuncs and functions from MDMC.mathematics)
    applied to UnitNDArray

    Parameterization:
    operation - function which is tested
    args - tuple of arguments which are applied to the function after the array
    """

    assert np.all(op(ARRAY, *args) == op(uarray, *args))


@pytest.mark.parametrize("cls", [UnitFloat, unit_array])
def test_init_None(cls):
    """
    Tests that passing None to a unit class (or helper function) returns None
    """

    assert cls(None, None) is None


def test_init_dtype():
    """
    Tests that the dtype can be initialized and set using UnitNDArray and
    unit_array
    """

    arr = np.array(LIST, dtype='object')
    uarr = UnitNDArray(len(LIST), UNIT, dtype='object')
    uarr[:] = LIST
    uarr_helper = unit_array(LIST, UNIT, dtype='object')

    def check_unit_array(unit_array):
        assert np.all(arr == uarr)
        assert arr.dtype == uarr.dtype

    check_unit_array(uarr)
    check_unit_array(uarr_helper)


def test_UnitNDArray_creation():
    """
    Tests that UnitNDArray can be created in the same three ways as ndarray:

    - explicit constructor call
    - view casting
    - new from template (e.g. slicing)
    """

    constrct = UnitNDArray(len(LIST), UNIT)
    constrct[:] = LIST

    vcast = ARRAY.view(UnitNDArray)
    vcast.unit = UNIT

    templte = unit_array(LIST+LIST, UNIT)[:len(LIST)]

    assert np.all(constrct == vcast)
    assert np.all(constrct == templte)
    assert constrct.unit == vcast.unit == templte.unit


def test_unit_array(uarray, uarray_helper):
    """
    Tests that the helper function unit_array returns the correct UnitNDArray
    """

    assert np.all(uarray == uarray_helper)
    assert uarray.unit == uarray_helper.unit
    assert uarray.dtype == uarray_helper.dtype
