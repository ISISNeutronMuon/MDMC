"""Tests for classes and functions in interaction_functions.py"""

from math import ceil

import pytest

from MDMC.common.units import Unit, UnitFloat
from MDMC.MD.interaction_functions import (Buckingham, Coulomb,
                                           HarmonicPotential,
                                           InteractionFunction, LennardJones,
                                           Periodic, Parameter,
                                           filter_parameters,
                                           filter_parameters_atom_attribute,
                                           filter_parameters_function,
                                           filter_parameters_interaction,
                                           filter_parameters_name,
                                           filter_parameters_structure,
                                           filter_parameters_value)
from MDMC.MD.simulation import Universe
from MDMC.MD.structural_units import Atom, Bond, Coulombic, Dispersion, Molecule


BUCK_A, BUCK_B, BUCK_C = 1., 2., 3.
BUCK_A_UNIT = Unit('kJ') / Unit('mol')
BUCK_B_UNIT = Unit('Ang') ** -1
BUCK_C_UNIT = Unit('Ang') ** 6 * Unit('kJ') / Unit('mol')
COULOMB_CHARGE = 5.0
COULOMB_CHARGE_UNIT = Unit('e')
HARMPOT_EQUIL_STATE, HARMPOT_POT_STREN = 10., 100.
HARMPOT_EQUIL_STATE_BOND_UNIT = Unit('Ang')
HARMPOT_POT_STREN_BOND_UNIT = Unit('kJ') / (Unit('mol') * Unit('Ang') ** 2)
HARMPOT_EQUIL_STATE_ANGLE_UNIT = Unit('deg')
HARMPOT_POT_STREN_ANGLE_UNIT = Unit('kJ') / (Unit('mol') * Unit('deg') ** 2)
LJ_EPSILON, LJ_SIGMA = 15., 5.
LJ_EPSILON_UNIT = Unit('kJ') / Unit('mol')
LJ_SIGMA_UNIT = Unit('Ang')
K1, K2, K3, K4 = 1., 2., 3., 4.
N1, N2, N3, N4 = 5, 6, 7, 8
D1, D2, D3, D4 = 9., 10., 11., 12.
NAME = 'length'
UNIT = Unit('Ang')
VALUE = 1.0
VAL_DICT = {'aa': UnitFloat(5, 'arb'),
            'bb': UnitFloat(7, 'arb'),
            'cc': UnitFloat(9, 'arb')}

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
def param_inter(parameter, coulombic):

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
    list
        A list of parameters with a value and a name. In each case the value is
        equal to the index of the parameter
    """

    return [Parameter(UnitFloat(VALUE * i, UNIT), NAME) for i in range(10)]

@pytest.fixture
def interaction_func():

    """
    Returns
    -------
    InteractionFunction
        An InteractionFunction object initialized with a dictionary containing
        name:value pairs that correspond to the parameters.
    """

    return InteractionFunction(VAL_DICT)

@pytest.fixture
def buckingham():

    """
    Returns
    -------
    Buckingham
        An Buckingham InteractionFunction initialized with the Buckingham
        A, B, and C parameters.
    """

    return Buckingham(BUCK_A, BUCK_B, BUCK_C)

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

    return Coulombic(atom_types=[1], universe=Universe(1.0), function=coulomb)

@pytest.fixture
def harmonic():

    """
    Returns
    -------
    HarmonicPotential
        A HarmonicPotential InteractionFunction initialized with an equilibrium
        state and a linear potential strength.
    """

    return HarmonicPotential(HARMPOT_EQUIL_STATE, HARMPOT_POT_STREN,
                             interaction_type='bond')

@pytest.fixture
def lennardjones():

    """
    Returns
    -------
    LennardJones
        A LennardJones InteractionFunction initialized with an LJ epsilon and
        LJ sigma values.
    """

    return LennardJones(LJ_EPSILON, LJ_SIGMA)

@pytest.fixture
def periodic():

    """
    Returns
    -------
    Periodic
        A Periodic InteractionFunction initialized with four order of parameter
        values e.g. K1, n1, d1, K2, n2, d2, K3, n3, d3, K4, n4, d4
    """

    return Periodic(K1, N1, D1, K2, N2, D2, K3, N3, D3, K4, N4, D4)


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


def test_interaction_setting_name(param_inter):

    """
    Tests that an error is raised when setting an interaction with a different
    name to interactions already in Parameter.interaction
    """

    with pytest.raises(ValueError):
        param_inter.interactions = Dispersion(Universe(1.0), [1, 1],
                                              function=coulomb)


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

    params = []
    for index in range(10):
        param = Parameter(VALUE * index, NAME, unit=UNIT)
        if index % 2:
            setattr(param, attr, val)
        params.append(param)

    assert filter_parameters(params, pred) == params[1::2]


@pytest.mark.parametrize('name, number', [('charge', 3),
                                          ('epsilon', 2),
                                          ('sigma', 0)])
def test_filter_parameters_name(name, number):

    """
    Tests that filtering parameters by name results in the correct number of
    parameters which have the correct name
    """

    params = [Parameter(VALUE * index, 'charge', unit=UNIT) if index < 3
              else Parameter(VALUE * index, 'epsilon', unit=UNIT)
              for index in range(5)]

    filtered = filter_parameters_name(params, name)
    assert [param.name for param in filtered] == [name] * number


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

    assert (filter_parameters_value(parameters, comp, value)
            == parameters[slice(*expected_slice)])


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

    for index, param in enumerate(parameters):
        if index % 2:
            param.interactions = coulombic
        else:
            param.interactions = Dispersion(Universe(1.0), [1, 1],
                                            function=LennardJones((1., 'arb'),
                                                                  (1., 'arb')))

    assert (filter_parameters_interaction(parameters, int_name)
            == parameters[slice(*expected_slice)])


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

    for index, param in enumerate(parameters):
        if index % 2:
            function = LennardJones((1., 'arb'), (1., 'arb'))
        else:
            function = coulomb
        param.interactions = Dispersion(Universe(1.0), [1, 1],
                                        function=function)

    assert (filter_parameters_function(parameters, function_name)
            == parameters[slice(*expected_slice)])


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
    inters = [Bond(Atom('H', mass=props[0], charge=props[1]),
                   Atom('H', mass=(2 * props[0]), charge=(props[1] * 2)))
              for props in [(1., 0.5),
                            (4., -1.0)]]

    for index, param in enumerate(parameters):
        # Set parameters with different interactions
        # So all parameters will have a Bond with Atoms with masses of 1. and 2.
        # and charges of 0.5 and 1.0, while only parameters with even indexes
        # will have a Bond with Atoms with masses of 4. and 8. and charges of
        # -1.0 and -2.0
        for inter_index, inter in enumerate(inters):
            if not index % (inter_index + 1):
                param.interactions = inter

    # Test that filter returns expected atoms for both val and val * 2, as any
    # parameter of an atom with val must also be a parameter of an atom with
    # val * 2
    assert (filter_parameters_atom_attribute(parameters, attr, val)
            == parameters[slice(*expected_slice)]
            == filter_parameters_atom_attribute(parameters, attr, val * 2))


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
    H2_bond = Bond(H2.atom_list[0], H2.atom_list[1])
    C_bond = Bond(Atom('C'), Atom('C'))

    for index, param in enumerate(parameters):
        if not index % 3:
            param.interactions = H2_bond
        if not index % 2:
            param.interactions = C_bond

    assert (filter_parameters_structure(parameters, struct_name)
            == parameters[slice(*expected_slice)])


def test_interaction_function_get_params(interaction_func):

    """
    Tests that the correct parameters are returned when retrieving them from
    an already-initialized InteractionFunction object.
    """

    for param in interaction_func.params:
        assert param.value == VAL_DICT[param.name]


def test_interaction_function_set_params(interaction_func, parameters):

    """
    Tests that the parameters of an InteractionFunction can be set.
    """

    interaction_func.params = parameters
    for intfunc_param, param in zip(interaction_func.params, parameters):
        assert intfunc_param.value == param.value


def test_interaction_function_params_values(interaction_func):

    """
    Tests that retrieval of the values of the parameters set during
    initialization of an InteractionFunction object returns the correct values.
    """

    assert all(interaction_func.params_values
               == [param.value for param in interaction_func.params])


def test_interaction_function_name(interaction_func):

    """
    Tests that the initialization of an InteractionFunction object has the
    correct name.
    """

    assert interaction_func.name == 'InteractionFunction'


@pytest.mark.filterwarnings("ignore: Coulombic")
def test_interaction_function_set_params_inters(interaction_func, coulombic):

    """
    Tests that the parent interaction for all Parameters of the
    InteractionFunction object can be set to a Coulombic Interaction object.
    """

    interaction_func.set_params_interactions(coulombic)
    for param in interaction_func.params:
        for inter in param.interactions:
            assert isinstance(inter, Coulombic)


@pytest.mark.parametrize("obj, values, names",
                         [(buckingham(), [BUCK_A, BUCK_B, BUCK_C],
                           ['A', 'B', 'C']),
                          (coulomb(), [COULOMB_CHARGE], ['charge']),
                          (harmonic(), [HARMPOT_EQUIL_STATE, HARMPOT_POT_STREN],
                           ['equilibrium_state', 'potential_strength']),
                          (lennardjones(), [LJ_EPSILON, LJ_SIGMA],
                           ['epsilon', 'sigma']),
                          (periodic(),
                           [K1, K2, K3, K4, D1, D2, D3, D4, N1, N2, N3, N4],
                           ['K1', 'K2', 'K3', 'K4', 'd1', 'd2', 'd3', 'd4',
                            'n1', 'n2', 'n3', 'n4'])])
def test_interaction_function_subclass_params(obj, values, names):

    """
    Tests that initializing a subclass of InteractionFunction assigns the
    correct values and names to the parameters.
    """

    for idx, param in enumerate(obj.params):
        assert param.value == values[idx]
        assert param.name == names[idx]


@pytest.mark.parametrize("inter_func_fixture, params",
                         [('buckingham', ['A', 'B', 'C']),
                          ('coulomb', ['charge']),
                          ('harmonic', ['equilibrium_state',
                                        'potential_strength']),
                          ('lennardjones', ['epsilon', 'sigma']),
                          ('periodic', ['K1', 'n1', 'd1', 'K2', 'n2', 'd2',
                                        'K3', 'n3', 'd3', 'K4', 'n4', 'd4'])])
def test_interaction_function_attributes(inter_func_fixture, params, request):

    """
    Tests that initializing a subclass of InteractionFunction creates an
    attribute with the name of each Parameter passed to __init__

    For example, a LennardJones object should have attributes epsilon and
    sigma, with a value of the corresponding Parameters
    """

    inter_func = request.getfixturevalue(inter_func_fixture)
    for param in params:
        # Test both for existence of attribute and that the Parameter has the
        # correct name
        assert hasattr(inter_func, param)
        assert getattr(inter_func, param).name == param


@pytest.mark.parametrize("inter_func_fixture, units",
                         [('buckingham', {'A':BUCK_A_UNIT,
                                          'B':BUCK_B_UNIT,
                                          'C':BUCK_C_UNIT}),
                          ('coulomb', {'charge':COULOMB_CHARGE_UNIT}),
                          ('lennardjones', {'epsilon':LJ_EPSILON_UNIT,
                                            'sigma':LJ_SIGMA_UNIT})])
def test_interaction_function_units(inter_func_fixture, units, request):

    """
    Tests that the units of the parameters of all subclasses of
    InteractionFunction (except HarmonicPotential) are set correctly
    """

    inter_func = request.getfixturevalue(inter_func_fixture)
    for param_name, unit in units.items():
        assert getattr(inter_func, param_name).unit == unit
        # Test an incorrect unit
        assert getattr(inter_func, param_name).unit != Unit('DOES_NOT_EXIST')


@pytest.mark.parametrize("inter_type, units",
                         [('bond', [HARMPOT_EQUIL_STATE_BOND_UNIT,
                                    HARMPOT_POT_STREN_BOND_UNIT]),
                          ('BoNd', [HARMPOT_EQUIL_STATE_BOND_UNIT,
                                    HARMPOT_POT_STREN_BOND_UNIT]),
                          ('angle', [HARMPOT_EQUIL_STATE_ANGLE_UNIT,
                                     HARMPOT_POT_STREN_ANGLE_UNIT]),
                          ('improper', [HARMPOT_EQUIL_STATE_ANGLE_UNIT,
                                        HARMPOT_POT_STREN_ANGLE_UNIT])])
def test_harmonic_potential_units(inter_type, units):

    """
    Tests that the units of the parameters of HarmonicPotential are set
    correctly, dependent on the interaction_type that is passed to it
    """

    h_pot = HarmonicPotential(1.0, 2.0, interaction_type=inter_type)
    # Ignore pylint warning for no member as both equilibrium_state and
    # potential_strength are created dynamically
    #pylint: disable=no-member
    assert h_pot.equilibrium_state.unit == units[0]
    assert h_pot.potential_strength.unit == units[1]


def test_harmonic_potential_invalid_inter_type():

    """
    Tests that if an invalid interaction_type is passed to HarmonicPotential, it
    raises a ValueError
    """

    with pytest.raises(ValueError):
        HarmonicPotential(5.0, 4.0, interaction_type='Bonded')


def test_harmonic_potential_no_inter_type():

    """
    Tests that if an interaction_type is no passed to HarmonicPotential, it
    raises a TypeError
    """

    with pytest.raises(TypeError):
        HarmonicPotential(6.0, 7.0, inter_tye='bond')


@pytest.mark.parametrize("params",
                         [(3., 1, 2.),
                          (5., 1, -30., 7., 3, 45.),
                          (9., 3, -40., 20., 4, -45., 60., 1, 9.),
                          (5., 1, 0.5, 7., 3, 8., 9., 0, 7.5, 4., 1, 9.9)])
def test_periodic_init(params):

    """
    Tests that initializing a Periodic InteractionFunction of different orders
    (first, second, third, and fourth) produces the expected parameters

    Tests that parameters are assigned the correct names and values
    """

    period = Periodic(*params)
    for index, param in enumerate(params, start=1):
        order = ceil(index / 3.)
        mod3_index = (order * 3)
        if mod3_index == 1:
            assert getattr(period, 'K{0}'.format(order)).value == param
        elif mod3_index == 2:
            assert getattr(period, 'n{0}'.format(order)).value == param
        elif mod3_index == 13:
            assert getattr(period, 'd{0}'.format(order)).value == param


@pytest.mark.parametrize("params",
                         [(1., ),
                          (3., 1),
                          (2., 9, 7., 4.),
                          (5., 1, -30., 7., 3),
                          (2., 9, 5., 2., 3, 9., 5),
                          (9., 3, -40., 20., 4, -45., 60., 1),
                          (4., 2, 10., 1., 1, -60., 2., 2, 90., 10.),
                          (5., 1, 0.5, 7., 3, 8., 9., 0, 7.5, 4., 1),
                          (5., 1, 0.5, 7., 3, 8., 9., 0, 7.5, 4., 1, 19., 10.)])
def test_periodic_invalid_num_parameters(params):

    """
    Tests that initializing a Periodic InteractionFunction with the incorrect
    number of parameters (i.e. not a multiple of 3), raises a TypeError

    Tests all numbers of parameters up to 13 (inclusive), except multiples of 3
    """

    with pytest.raises(TypeError):
        Periodic(*params)


@pytest.mark.parametrize("params",
                         [(3., 1.2, 2.),
                          (5., 1, -30., 7., 3., 45.),
                          (9., 3, -40., 20., 4, -45., 60., 7.2, 9.),
                          (5., 1, 0.5, 7., 3, 8., 9., 0, 7.5, 4., 1.6, 9.9),
                          (5., 1., 0.5, 7., 3., 8., 9., 0., 7., 4., 1., 9.9)])
def test_periodic_init_types(params):

    """
    Tests that initializing a Periodic InteractiongFunction with an n value (of
    any order) which is not an int, raises a TypeError
    """

    with pytest.raises(TypeError):
        Periodic(*params)

@pytest.mark.parametrize("params",
                         [(3., -1, 2.),
                          (5., 1, -30., 7., -3, 45.),
                          (9., 3, -40., 20., 4, -45., 60., -7, 9.),
                          (5., 1, 0.5, 7., 3, 8., 9., 0, 7.5, 4., -1, 9.9),
                          (5., -1, 0.5, 7., -3, 8., 9., -10, 7., 4., -1, 9.9)])
def test_periodic_init_values(params):

    """
    Tests that initializing a Periodic InteractiongFunction with an n value (of
    any order) which is negative, raises a ValueError
    """

    with pytest.raises(ValueError):
        Periodic(*params)
