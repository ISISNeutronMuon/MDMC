"""Tests for classes and functions in interaction_functions.py"""

import pytest

from MDMC.common.units import Unit, UnitFloat
from MDMC.MD.interaction_functions import Parameter, Coulomb, LennardJones
from MDMC.MD.simulation import Universe
from MDMC.MD.structural_units import Atom, Coulombic, Dispersion


VALUE = 1.0
NAME = 'length'
UNIT = Unit('Ang')
COULOMB = Coulomb((5.0, 'e'))
COULOMBIC = Coulombic(atom_types=[1],
                      universe=Universe(1.0),
                      function=COULOMB)

@pytest.fixture
def parameter():

    """
    Returns
    -------
    Parameter
        A Parameter with a value and a name
    """

    return Parameter(UnitFloat(VALUE, UNIT), NAME)


@pytest.fixture
def scaled_parameter():

    """
    Returns
    -------
    Parameter
        A Parameter with a scaled value and a name
    """

    return Parameter(UnitFloat(5 * VALUE, UNIT), NAME)


@pytest.fixture
def param_inter(parameter):

    """
    Returns
    -------
    Parameter
        A Parameter with a value, a name, and an interaction
    """

    parameter.interactions = COULOMBIC
    return parameter


@pytest.mark.parametrize('value, unit', [(VALUE, UNIT),
                                         (UnitFloat(VALUE, UNIT), None)])
def test_parameter_value_init(value, unit):

    """
    Tests that Parameters can be initialised by either passing a UnitFloat as
    the value, or by passing a value and a unit
    """

    param = (Parameter(value, NAME, unit=unit) if unit
             else Parameter(value, NAME))
    assert param.value == VALUE
    assert param.unit == UNIT


def test_tied_parameters(parameter, scaled_parameter):

    """
    Tests that parameters that are tied with a scale of 1 have the same value,
    including after the value of the tie parameter is changed
    """

    assert scaled_parameter.value != parameter.value

    scaled_parameter.set_tie(parameter, "* 1")
    assert scaled_parameter.value == parameter.value

    parameter.value *= 2.
    assert scaled_parameter.value == parameter.value


def test_tied_parameter_change_warning(parameter, scaled_parameter):

    """
    Tests that parameters that are tied issue a warning when the value is set,
    and that the value does not change
    """

    scaled_parameter.set_tie(parameter, "* 1")
    with pytest.warns(UserWarning):
        scaled_parameter.value = parameter.value * 10.
    assert scaled_parameter.value == parameter.value


def test_parameter_tied(parameter, scaled_parameter):

    """
    Tests that tying a Parameter changes tied attribute from False to True
    """

    assert not scaled_parameter.tied
    scaled_parameter.set_tie(parameter, "* 1")
    assert scaled_parameter.tied


def test_fixed_parameter_change_warning(parameter):

    """
    Tests that parameters that are fixed issue a warning when the value is set,
    and that the value does not change
    """

    parameter.fixed = True
    with pytest.warns(UserWarning):
        parameter.value *= 5.
    assert parameter.value == VALUE


@pytest.mark.parametrize('constraints, value', [((0., 2.), 2.),
                                                ((0., 2.), 0.),
                                                ((1., 5.), 2.),
                                                ((-1., 2.), -0.5)])
def test_value_setting_within_constraints(constraints, value, parameter):

    """
    Tests setting the value of a Parameter within the constraints

    Includes tests of values at edges of constraints, as constraints are a
    closed interval
    """

    parameter.constraints = constraints
    parameter.value = value
    assert parameter.value == value


@pytest.mark.parametrize('constraints, value', [((1., 2.), 0.),
                                                ((1., 5.), 6.),
                                                ((-1., 2.), -1.5)])
def test_value_setting_outside_constraints(constraints, value, parameter):

    """
    Tests that setting the value of a Parameter outside of the constraints
    raises an error, and that the Parameter value does not change

    Includes setting the value both above and below the constraints
    """

    parameter.constraints = constraints
    with pytest.raises(ValueError):
        parameter.value = value
    assert parameter.value == VALUE


def test_interaction_setting_name(param_inter):

    """
    Tests that an error is raised when setting an interaction with a different
    name to interactions already in Parameter.interaction
    """

    with pytest.raises(ValueError):
        param_inter.interactions = Dispersion(Universe(1.0), [1],
                                              function=COULOMB)


def test_interaction_setting_function_name(param_inter):

    """
    Tests that an error is raised when setting an interaction with an
    interaction function with a different name to the interaction functions of
    interactions already in Parameter.interaction
    """

    with pytest.raises(ValueError):
        param_inter.interactions = Coulombic(Universe(1.0), atom_types=[1],
                                             function=LennardJones((1., 'arb'),
                                                                   (1., 'arb')))
