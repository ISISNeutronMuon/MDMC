"""Tests related to the Unit class

AUTHOR :    Thomas Farmer        START DATE :    18/12/2018, 16:02:17"""

import numpy as np
import pytest

from MDMC.common.units import Unit


STRING = 'kg'


@pytest.fixture
def unit():

    return Unit(STRING)


def test_subclass_string(unit):

    """
    Tests that Unit objects have str as base class
    """

    assert isinstance(unit, str)
    assert not isinstance(unit, float)


@pytest.mark.parametrize("op,args", [('capitalize', ()),
                                     ('split', ()),
                                     ('islower', ()),
                                     ('find', ('g', )),
                                     ('replace', ('k', 'm'))])
def test_string_operations(unit, op, args):

    """
    Tests that common string operations are also valid for Unit objects
    """

    assert getattr(unit, op)(*args) == getattr(STRING, op)(*args)
    assert getattr(unit, op)(*args) != getattr('Pa', op)(*args)


def test_multiply_Unit(unit):

    """
    Tests that * operation has the expected behaviour
    """

    multiply = unit * unit
    assert multiply == STRING + ' ' + STRING
    assert isinstance(multiply, Unit)
    with pytest.raises(TypeError):
        invalid = unit * 2


def test_divide_Unit(unit):

    """
    Tests that / operation has the expected behaviour
    """

    divide = unit / unit
    assert divide == STRING + ' / ' + STRING
    assert isinstance(divide, Unit)
    with pytest.raises(TypeError):
        invalid = unit / 2


@pytest.mark.parametrize("input,expected", [(2, ' ^ 2'),
                                            (2.0, ' ^ 2.0'),
                                            (np.float64(2.), ' ^ 2.0'),
                                            (unit(), TypeError),
                                            (STRING, TypeError),
                                            ('2', ' ^ 2.0')])
def test_power_Unit(unit, input, expected):

    """
    Tests that ** operation has the expected behaviour

    Non-numeric types (or types that cannot be cast to float) should raise a
    TypeError
    """

    if callable(expected) and isinstance(expected(), Exception):
        with pytest.raises(expected):
            invalid = unit ** input
    else:
        power = unit ** input
        assert power == STRING + expected
        assert isinstance(power, Unit)
