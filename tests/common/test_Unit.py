"""Tests related to the Unit class"""

import numpy as np
import pytest
from pytest_cases import fixture_ref

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
    assert multiply == STRING + ' ^ 2'
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
                                            (2.0, ' ^ 2'),
                                            (np.float64(2.), ' ^ 2'),
                                            (fixture_ref(unit), TypeError),
                                            (STRING, TypeError),
                                            ('2', ' ^ 2')])
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


@pytest.mark.parametrize("input, base, numerator, denominator",
                         [('Ang', True, ['Ang'], []),
                          ('Ang mol', False, ['Ang', 'mol'], []),
                          ('Ang ^ 3', False, ['Ang', 'Ang', 'Ang'], []),
                          ('Ang^3', False, ['Ang', 'Ang', 'Ang'], []),
                          ('Ang^ 3', False, ['Ang', 'Ang', 'Ang'], []),
                          ('Ang ^3', False, ['Ang', 'Ang', 'Ang'], []),
                          ('Ang / mol', False, ['Ang'], ['mol']),
                          ('e^2 K J^2', False, ['e', 'e', 'K', 'J', 'J'], []),
                          ('e^2 K / J^2', False, ['e', 'e', 'K'], ['J', 'J'])])
def test_determine_components(input, base, numerator, denominator):
    """
    Tests that the numerator and denominator components of a Unit are correctly
    determined upon passing a string, and whether or not the Unit is base
    """

    t_unit = Unit(input)
    assert t_unit.base == base
    assert t_unit.components['numerator'] == numerator
    assert t_unit.components['denominator'] == denominator


@pytest.mark.parametrize("string, numerator, denominator",
                         [('Ang ^ -3', [], ['Ang', 'Ang', 'Ang']),
                          ('Ang^-3', [], ['Ang', 'Ang', 'Ang']),
                          ('Ang^ -3', [], ['Ang', 'Ang', 'Ang']),
                          ('Ang ^-3', [], ['Ang', 'Ang', 'Ang']),
                          ('Ang^-1 / mol^-1', ['mol'], ['Ang']),
                          ('e^-2 K J^-2', ['K'], ['e', 'e', 'J', 'J']),
                          ('e^-2 K / J^-2', ['K', 'J', 'J'], ['e', 'e'])])
def test_negative_powers(string, numerator, denominator):
    """
    Tests that the numerator and denominator components of a Unit are correctly
    determined upon passing a string that contains negative powers
    """

    t_unit = Unit(string)
    assert t_unit.components['numerator'] == numerator
    assert t_unit.components['denominator'] == denominator


@pytest.mark.parametrize("string, numerator, denominator",
                         [('(Ang)', ['Ang'], []),
                          ('[Ang]', ['Ang'], [])])
def test_unsupported_characters(string, numerator, denominator):
    """
    Tests that the numerator and denominator components of a Unit are correctly
    determined upon passing a string that contains unsupported characters
    """

    t_unit = Unit(string)
    assert t_unit.components['numerator'] == numerator
    assert t_unit.components['denominator'] == denominator


@pytest.mark.parametrize("string, conversion_factor",
                         [('Ang', 1.),
                          ('nm', 10.),
                          ('1 / nm', 0.1),
                          ('10^3 nm', 10000.),
                          ('10^3 nm / 10^-3 cm', 0.1)])
def test_conversion_factor(string, conversion_factor):
    """
    Tests that the ``conversion_factor`` of a Unit is correctly determined upon
    passing a string
    """

    t_unit = Unit(string)
    assert t_unit.conversion_factor == conversion_factor


@pytest.mark.parametrize("string, physical_property",
                         [('Ang', 'LENGTH'),
                          ('nm', 'LENGTH'),
                          ('fs', 'TIME'),
                          ('s', 'TIME'),
                          ('amu', 'MASS'),
                          ('kg', 'MASS'),
                          ('kJ / mol', 'ENERGY'),
                          ('kcal', 'ENERGY'),
                          ('kJ / Ang mol', 'FORCE'),
                          ('kcal / Ang mol', 'FORCE'),
                          ('Pa', 'PRESSURE'),
                          ('atm', 'PRESSURE'),
                          ('deg', 'ANGLE'),
                          ('rad', 'ANGLE')])
def test_physical_property(string, physical_property):
    """
    Tests that the ``physical_property`` of a Unit is correctly determined upon
    passing a string
    """

    t_unit = Unit(string)
    assert t_unit.physical_property == physical_property
