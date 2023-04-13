"""Tests for classes in MDMC.MD.parameters"""

import pytest

from MDMC.common.units import Unit, UnitFloat
from MDMC.MD.interaction_functions import Coulomb, LennardJones
from MDMC.MD.parameters import Parameter, Parameters
from MDMC.MD.simulation import Universe
from MDMC.MD.structures import Atom, Molecule
from MDMC.MD.interactions import Bond, Dispersion, Coulombic

NAME = 'length'
UNIT = Unit('Ang')
VALUE = 1.0
COULOMB_CHARGE = 5.0


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
def coulomb():
    """
    Returns
    -------
    Coulomb
        A Coulomb InteractionFunction initialized with a charge parameter.
    """

    return Coulomb(COULOMB_CHARGE)

@pytest.fixture
def coulombic(coulomb):
    """
    Returns
    -------
    Coulombic
        A Coulombic Interaction object, initialized with a Coulomb
        InteractionFunction object, an empty universe, and one atom.
    """

    return Coulombic(atom_types=[1], universe=Universe(1.0, verbose=False), function=coulomb)

@pytest.fixture
def parameter_inter(parameter, coulombic):
    """
    Returns
    -------
    Parameter
        A Parameter with a value, a name, and an interaction
    """

    parameter.interactions = coulombic
    return parameter

@pytest.fixture
def parameters():
    """
    Returns
    -------
    Parameters
        A Parameters object of Parameter objects with a value and a name. In
        each case the value is equal to the index of the parameter
    """

    return Parameters([Parameter(UnitFloat(VALUE * i, UNIT), NAME + str(i)) for i
                       in range(10)])


@pytest.mark.parametrize('value, unit', [(VALUE, UNIT),
                                         (UnitFloat(VALUE, UNIT), None)])
def test_parameter_value_init(value, unit):
    """
    Tests that Parameters can be initialised by either passing a UnitFloat as
    the value, or by passing a value and a unit
    """

    parameter = (Parameter(value, NAME, unit=unit) if unit
             else Parameter(value, NAME))
    assert parameter.value == VALUE
    assert parameter.unit == UNIT


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
    Tests that the value of a Parameter can be set within the constraints

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


def test_interaction_setting_name(parameter_inter, coulomb):
    """
    Tests that an error is raised when setting an interaction with a different
    name to interactions already in Parameter.interaction
    """

    with pytest.raises(ValueError):
        parameter_inter.interactions = Dispersion(Universe(1.0, verbose=False), [1, 1],
                                              function=coulomb)


def test_interaction_setting_function_name(parameter_inter):
    """
    Tests that an error is raised when setting an interaction with an
    interaction function with a different name to the interaction functions of
    interactions already in Parameter.interaction
    """

    with pytest.raises(ValueError):
        parameter_inter.interactions = Coulombic(Universe(1.0, verbose=False), atom_types=[1],
                                             function=LennardJones((1., 'arb'),
                                                                   (1., 'arb')))


@pytest.mark.parametrize('expression, expected', [('*2.', VALUE * 2.),
                                                  ('/2.', VALUE / 2.),
                                                  ('+2.', VALUE + 2.),
                                                  ('-2.', VALUE - 2.)])
def test_parameter_set_tie(expression, expected, parameter, scaled_parameter):
    """
    Tests setting Parameter tie with basic arithmetic operations
    """

    scaled_parameter.set_tie(parameter, expression)
    assert scaled_parameter.value == expected


@pytest.mark.parametrize('pred, attr, val', [(lambda p: p.fixed is True,
                                              'fixed',
                                              True),
                                             ((lambda p: p.constraints
                                               is not None),
                                              'constraints',
                                              (0., 10.)),
                                             (lambda p: p.unit is 'e',
                                              'unit',
                                              'e')])
def test_filter_parameters(pred, attr, val):
    """
    Tests parameter filtering for predicates not used by other Parameter filter
    functions
    """

    parameters = Parameters()
    for index in range(10):
        parameter = Parameter(VALUE * index, NAME + str(index), unit=UNIT)
        if index % 2:
            setattr(parameter, attr, val)
        parameters.append(parameter)

    expected_parameters = Parameters(list(parameters.values())[1::2])

    assert parameters.filter(pred) == expected_parameters


@pytest.mark.parametrize('name, number', [('charge', 3),
                                          ('epsilon', 2),
                                          ('sigma', 0)])
def test_filter_parameters_name(name, number):
    """
    Tests that filtering parameters by name results in the correct number of
    parameters which have the correct name
    """

    parameters = Parameters([Parameter(VALUE * index, 'charge', unit=UNIT)
                         if index < 3
                         else Parameter(VALUE * index, 'epsilon', unit=UNIT)
                         for index in range(5)])

    filtered = parameters.filter_name(name)
    assert [parameter.type for parameter in filtered.values()] == [name] * number


@pytest.mark.parametrize('comp, value, expected_slice', [('<', 0., [-1, -2]),
                                                         ('>=', 0., [0, None]),
                                                         ('>', 5., [6, None]),
                                                         ('<', 2., [0, 2]),
                                                         ('>=', 5., [5, None]),
                                                         ('<=', 2., [0, 3]),
                                                         ('==', 1., [1, 2]),
                                                         ('!=', 9., [0, -1])])
def test_filter_parameter_value(comp, value, expected_slice, parameters):
    """
    Tests that the filtering parameters by value results in the correct
    parameters being returned

    Tests all supported comparison operators. First parametrization tests case
    where no parameters are returned, second parameterization tests case where
    all parameters are returned.
    """
    expected_parameters = Parameters(list(parameters.values())[slice(*expected_slice)])

    assert parameters.filter_value(comp, value) == expected_parameters


@pytest.mark.parametrize('int_name, expected_slice', [('Dispersion',
                                                       [0, None, 2]),
                                                      ('Coulombic',
                                                       [1, None, 2]),
                                                      ('Bond',
                                                       [-1, -2])])
def test_filter_parameters_interaction(int_name, expected_slice, parameters,
                                       coulombic):
    """
    Tests that filtering parameters by interaction results in the correct
    parameters being returned
    """

    for index, parameter in enumerate(parameters.values()):
        if index % 2:
            parameter.interactions = coulombic
        else:
            parameter.interactions = Dispersion(Universe(1.0, verbose=False), [1, 1],
                                            function=LennardJones((1., 'arb'),
                                                                  (1., 'arb')))

    expected_parameters = Parameters(list(parameters.values())[slice(*expected_slice)])

    assert parameters.filter_interaction(int_name) == expected_parameters


@pytest.mark.parametrize('function_name, expected_slice', [('Coulomb',
                                                            [0, None, 2]),
                                                           ('LennardJones',
                                                            [1, None, 2]),
                                                           ('HarmonicPotential',
                                                            [-1, -2])])
def test_filter_parameters_function(function_name, expected_slice, parameters,
                                    coulomb):
    """
    Tests that filtering parameters by interaction function results in the
    correct number of parameters which have the correct interaction function
    """

    for index, parameter in enumerate(parameters.values()):
        if index % 2:
            function = LennardJones((1., 'arb'), (1., 'arb'))
        else:
            function = coulomb
        parameter.interactions = Dispersion(Universe(1.0, verbose=False), [1, 1],
                                        function=function)

    expected_parameters = Parameters(list(parameters.values())[slice(*expected_slice)])

    assert parameters.filter_function(function_name) == expected_parameters


@pytest.mark.filterwarnings("ignore: Coulombic")
@pytest.mark.parametrize('attr, val, expected_slice', [('mass', 1.,
                                                        [0, None]),
                                                       ('mass', 4.,
                                                        [0, None, 2]),
                                                       ('charge', .5,
                                                        [0, None]),
                                                       ('charge', -1.,
                                                        [0, None, 2])])
def test_filter_parameters_atom_attr(attr, val, expected_slice, parameters):
    """
    Tests that filtering parameters by the values of an attribute of the atoms
    which have the parameter applied to them results in the correct parameters
    being returned

    Tests for different charges and masses, and tests that all atoms that have
    the parameter applied to them are considered when filtering
    """

    # Make two bonds with atoms with different masses and charges
    # The second atom in each Bond has double the mass and charge of the first
    # atom
    inters = [Bond(Atom('H', mass=props[0], charge=props[1], cutoff=10.),
                   Atom('H', mass=(2 * props[0]), charge=(props[1] * 2), cutoff=10.))
              for props in [(1., 0.5),
                            (4., -1.0)]]

    for index, parameter in enumerate(parameters.values()):
        # Set parameters with different interactions
        # So all parameters will have a Bond with Atoms with masses of 1. and 2.
        # and charges of 0.5 and 1.0, while only parameters with even indexes
        # will have a Bond with Atoms with masses of 4. and 8. and charges of
        # -1.0 and -2.0
        for inter_index, inter in enumerate(inters):
            if not index % (inter_index + 1):
                parameter.interactions = inter

    expected_parameters = Parameters(list(parameters.values())[slice(*expected_slice)])

    # Test that filter returns expected atoms for both val and val * 2, as any
    # parameter of an atom with val must also be a parameter of an atom with
    # val * 2
    assert (parameters.filter_atom_attribute(attr, val)
            == expected_parameters
            == parameters.filter_atom_attribute(attr, val * 2))


@pytest.mark.parametrize('struct_name, expected_slice', [('H', [0, None, 3]),
                                                         ('C', [0, None, 2]),
                                                         ('H2', [0, None, 3])])
def test_filter_parameters_structure(struct_name, expected_slice, parameters):
    """
    Tests that filtering parameters by the structures which have the parameter
    applied to them results in the correct parameters being returned

    Tests for both atoms and molecules as athe structural unit
    """

    # Create bonds that can be set as the a parameter's interactions
    H2 = Molecule(atoms=[Atom('H'), Atom('H')], name='H2')
    H2_bond = Bond(H2.atoms[0], H2.atoms[1])
    C_bond = Bond(Atom('C'), Atom('C'))

    for index, parameter in enumerate(parameters.values()):
        if not index % 3:
            parameter.interactions = H2_bond
        if not index % 2:
            parameter.interactions = C_bond

    expected_parameters = Parameters(list(parameters.values())[slice(*expected_slice)])

    assert (parameters.filter_structure(struct_name)
            == expected_parameters)


def test_parameters_getitem_lazy():
    """Tests that the user can get a parameter without using its ID"""

    parameters = Parameters([Parameter(name='charge', value=1.),
                             Parameter(name='epsilon', value=2.),
                             Parameter(name='sigma', value=3.)])

    for test_parameter in [('charge', 1.), ('epsilon', 2.), ('sigma', 3.)]:
        assert parameters[test_parameter[0]].value == test_parameter[1]

    with pytest.raises(KeyError):
        _ = parameters['nonexistent_parameter']
