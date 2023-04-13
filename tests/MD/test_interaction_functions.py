"""Tests for classes in MDMC.MD.interaction_functions"""

from math import ceil

import numpy as np
import pytest
from pytest_cases import parametrize, fixture_ref

from MDMC.common.units import Unit, UnitFloat
from MDMC.MD.interaction_functions import (Buckingham, Coulomb,
                                           HarmonicPotential,
                                           InteractionFunction, LennardJones,
                                           Periodic)
from MDMC.MD.parameters import Parameter, Parameters
from MDMC.MD.simulation import Universe
from MDMC.MD.interactions import Coulombic

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
HARMPOT_POT_STREN_ANGLE_UNIT = Unit('kJ') / (Unit('mol') * Unit('rad') ** 2)
LJ_EPSILON, LJ_SIGMA = 15., 5.
LJ_EPSILON_UNIT = Unit('kJ') / Unit('mol')
LJ_SIGMA_UNIT = Unit('Ang')
K1, K2, K3, K4 = 1., 2., 3., 4.
N1, N2, N3, N4 = 5, 6, 7, 8
D1, D2, D3, D4 = 9., 10., 11., 12.
K_UNIT = Unit('kJ') / Unit('mol')
D_UNIT = Unit('deg')
NAME = 'length'
UNIT = Unit('Ang')
VALUE = 1.0
VAL_DICT = {'aa': UnitFloat(5, 'arb'),
            'bb': UnitFloat(7, 'arb'),
            'cc': UnitFloat(9, 'arb')}


@pytest.fixture
def parameters():
    """
    Returns
    -------
    Parameters
        A Parameters object of Parameter objecys with a value and a name. In
        each case the value is equal to the index of the parameter
    """

    return Parameters([Parameter(UnitFloat(VALUE * i, UNIT), NAME + str(i)) for i
                       in range(10)])

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

    return Coulombic(atom_types=[1], universe=Universe(1.0, verbose=False), function=coulomb)

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


def test_interaction_function_get_parameters(interaction_func):
    """
    Tests that the correct parameters are returned when retrieving them from
    an already-initialized InteractionFunction object.
    """

    for parameter in interaction_func.parameters.as_array:
        assert parameter.value == VAL_DICT[parameter.type]


def test_interaction_function_set_parameters(interaction_func, parameters):
    """
    Tests that the parameters of an InteractionFunction can be set.
    """

    interaction_func.parameters = parameters
    for intfunc_parameter, parameter in zip(interaction_func.parameters.values(), parameters.values()):
        assert intfunc_parameter.value == parameter.value


def test_interaction_function_parameters_values(interaction_func):
    """
    Tests that retrieval of the values of the parameters set during
    initialization of an InteractionFunction object returns the correct values.
    """

    assert all(interaction_func.parameters_values
               == [parameter.value for parameter in interaction_func.parameters.as_array])


def test_interaction_function_name(interaction_func):
    """
    Tests that the initialization of an InteractionFunction object has the
    correct name.
    """

    assert interaction_func.name == 'InteractionFunction'


@pytest.mark.filterwarnings("ignore: Coulombic")
def test_interaction_function_set_parameters_inters(interaction_func, coulombic):
    """
    Tests that the parent interaction for all Parameters of the
    InteractionFunction object can be set to a Coulombic Interaction object.
    """

    interaction_func.set_parameters_interactions(coulombic)
    for parameter in interaction_func.parameters.as_array:
        for inter in parameter.interactions:
            assert isinstance(inter, Coulombic)


@parametrize("obj, values, names",
                         [(fixture_ref(buckingham), [BUCK_A, BUCK_B, BUCK_C],
                           ['A', 'B', 'C']),
                          (fixture_ref(coulomb), [COULOMB_CHARGE], ['charge']),
                          (fixture_ref(harmonic), [HARMPOT_EQUIL_STATE, HARMPOT_POT_STREN],
                           ['equilibrium_state', 'potential_strength']),
                          (fixture_ref(lennardjones), [LJ_EPSILON, LJ_SIGMA],
                           ['epsilon', 'sigma']),
                          (fixture_ref(periodic),
                           [K1, K2, K3, K4, D1, D2, D3, D4, N1, N2, N3, N4],
                           ['K1', 'K2', 'K3', 'K4', 'd1', 'd2', 'd3', 'd4',
                            'n1', 'n2', 'n3', 'n4'])])
def test_interaction_function_subclass_parameters(obj, values, names):
    """
    Tests that initializing a subclass of InteractionFunction assigns the
    correct values and names to the parameters.
    """
    
    for value, name in zip(values, names):
        assert obj.parameters[name].value == value



@parametrize("inter_func, parameters",
                         [(fixture_ref(buckingham), ['A', 'B', 'C']),
                          (fixture_ref(coulomb), ['charge']),
                          (fixture_ref(harmonic), ['equilibrium_state',
                                        'potential_strength']),
                          (fixture_ref(lennardjones), ['epsilon', 'sigma']),
                          (fixture_ref(periodic), ['K1', 'n1', 'd1', 'K2', 'n2', 'd2',
                                        'K3', 'n3', 'd3', 'K4', 'n4', 'd4'])])
def test_interaction_function_attributes(inter_func, parameters, request):
    """
    Tests that initializing a subclass of InteractionFunction creates an
    attribute with the name of each Parameter passed to __init__

    For example, a LennardJones object should have attributes epsilon and
    sigma, with a value of the corresponding Parameters
    """

    for parameter in parameters:
        # Test both for existence of attribute and that the Parameter has the
        # correct type
        assert hasattr(inter_func, parameter)
        assert getattr(inter_func, parameter).type == parameter


@pytest.mark.parametrize("inter_func, units",
                         [(Buckingham(BUCK_A, BUCK_B, BUCK_C),
                           {'A':BUCK_A_UNIT,
                            'B':BUCK_B_UNIT,
                            'C':BUCK_C_UNIT}),
                          (Buckingham(A=BUCK_A, B=BUCK_B, C=BUCK_C),
                           {'A':BUCK_A_UNIT,
                            'B':BUCK_B_UNIT,
                            'C':BUCK_C_UNIT}),
                          (Buckingham(BUCK_A, BUCK_B, C=BUCK_C),
                           {'A':BUCK_A_UNIT,
                            'B':BUCK_B_UNIT,
                            'C':BUCK_C_UNIT}),
                          (Buckingham(BUCK_A, C=BUCK_C, B=BUCK_B),
                           {'A':BUCK_A_UNIT,
                            'B':BUCK_B_UNIT,
                            'C':BUCK_C_UNIT}),
                          (Coulomb(COULOMB_CHARGE),
                           {'charge':COULOMB_CHARGE_UNIT}),
                          (Coulomb(charge=COULOMB_CHARGE),
                           {'charge':COULOMB_CHARGE_UNIT}),
                          (LennardJones(LJ_EPSILON, LJ_SIGMA),
                           {'epsilon':LJ_EPSILON_UNIT,
                            'sigma':LJ_SIGMA_UNIT}),
                          (LennardJones(epsilon=LJ_EPSILON, sigma=LJ_SIGMA),
                           {'epsilon':LJ_EPSILON_UNIT,
                            'sigma':LJ_SIGMA_UNIT}),
                          (LennardJones(LJ_EPSILON, sigma=LJ_SIGMA),
                           {'epsilon':LJ_EPSILON_UNIT,
                            'sigma':LJ_SIGMA_UNIT})])
def test_interaction_function_units(inter_func, units):
    """
    Tests that the units of the parameters of all subclasses of
    InteractionFunction (except HarmonicPotential) are set correctly when using
    positional arguments, keyword arguments, and a mixture of the two
    """

    for parameter_name, unit in units.items():
        assert getattr(inter_func, parameter_name).unit == unit
        # Test an incorrect unit
        assert getattr(inter_func, parameter_name).unit != Unit('DOES_NOT_EXIST')


@pytest.mark.parametrize("inter_type, units",
                         [('bond', [HARMPOT_EQUIL_STATE_BOND_UNIT,
                                    HARMPOT_POT_STREN_BOND_UNIT]),
                          ('BoNd', [HARMPOT_EQUIL_STATE_BOND_UNIT,
                                    HARMPOT_POT_STREN_BOND_UNIT]),
                          ('angle', [HARMPOT_EQUIL_STATE_ANGLE_UNIT,
                                     HARMPOT_POT_STREN_ANGLE_UNIT]),
                          ('BondAngle', [HARMPOT_EQUIL_STATE_ANGLE_UNIT,
                                         HARMPOT_POT_STREN_ANGLE_UNIT]),
                          ('improper', [HARMPOT_EQUIL_STATE_ANGLE_UNIT,
                                        HARMPOT_POT_STREN_ANGLE_UNIT])])
def test_harmonic_potential_units(inter_type, units):
    """
    Tests that the units of the parameters of HarmonicPotential are set
    correctly, dependent on the interaction_type that is passed to it, for
    positional, mixed, and keyword assignment.
    """

    h_pot_list = []
    h_pot_list.append(HarmonicPotential(1.0, 2.0, interaction_type=inter_type))
    h_pot_list.append(HarmonicPotential(1.0, potential_strength=2.0,
                                        interaction_type=inter_type))
    h_pot_list.append(HarmonicPotential(equilibrium_state=1.0,
                                        potential_strength=2.0,
                                        interaction_type=inter_type))
    # Ignore pylint warning for no member as both equilibrium_state and
    # potential_strength are created dynamically
    #pylint: disable=no-member
    for h_pot in h_pot_list:
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
    Tests that if a `bond` interaction_type is passed to HarmonicPotential, it
    raises a TypeError
    """

    with pytest.raises(TypeError):
        HarmonicPotential(6.0, 7.0, inter_tye='bond')


@pytest.mark.parametrize("parameters",
                         [(3., 1, 2.),
                          (5., 1, -30., 7., 3, 45.),
                          (5., np.int64(1), -30., 7., 3, 45.),
                          (5., 1, -30., 7., np.int32(3), 45.),
                          (9., 3, -40., 20., 4, -45., 60., 1, 9.),
                          (5., 1, 0.5, 7., 3, 8., 9., 0, 7.5, 4., 1, 9.9),
                          {'K1':3., 'n1':1, 'd1':2.}])
def test_periodic_init(parameters):
    """
    Tests that initializing a Periodic InteractionFunction of different orders
    (first, second, third, and fourth) produces the expected parameters

    Tests that parameters are assigned the correct names, values and units

    The third and fourth parametrizations test that numpy integers can also be
    used for specifying the n parameters

    Test that for the first order, keyword assignment of arguments works. For
    higher orders, arguments must be provided positionally.
    """

    if isinstance(parameters, tuple):
        period = Periodic(*parameters)
    else:
        period = Periodic(**parameters)
        parameters = parameters.values()

    for index, parameter in enumerate(parameters, start=1):
        order = ceil(index / 3.)
        # index % 3 determines whether the parameter is K, n or d
        mod3_index = (index % 3)
        if mod3_index == 1:
            assert getattr(period, 'K{0}'.format(order)).value == parameter
            assert getattr(period, 'K{0}'.format(order)).unit == K_UNIT
        elif mod3_index == 2:
            assert getattr(period, 'n{0}'.format(order)).value == parameter
            # n is unitless
            assert getattr(period, 'n{0}'.format(order)).unit is None
        elif mod3_index == 0:
            assert getattr(period, 'd{0}'.format(order)).value == parameter
            assert getattr(period, 'd{0}'.format(order)).unit == D_UNIT


@pytest.mark.parametrize("parameters",
                         [(1., ),
                          (3., 1),
                          (2., 9, 7., 4.),
                          (5., 1, -30., 7., 3),
                          (2., 9, 5., 2., 3, 9., 5),
                          (9., 3, -40., 20., 4, -45., 60., 1),
                          (4., 2, 10., 1., 1, -60., 2., 2, 90., 10.),
                          (5., 1, 0.5, 7., 3, 8., 9., 0, 7.5, 4., 1),
                          (5., 1, 0.5, 7., 3, 8., 9., 0, 7.5, 4., 1, 19., 10.)])
def test_periodic_invalid_num_parameters(parameters):
    """
    Tests that initializing a Periodic InteractionFunction with the incorrect
    number of parameters (i.e. not a multiple of 3), raises a TypeError

    Tests all numbers of parameters up to 13 (inclusive), except multiples of 3
    """

    with pytest.raises(TypeError):
        Periodic(*parameters)


@pytest.mark.parametrize("parameters",
                         [(3., 1.2, 2.),
                          (5., 1, -30., 7., 3., 45.),
                          (9., 3, -40., 20., 4, -45., 60., 7.2, 9.),
                          (5., 1, 0.5, 7., 3, 8., 9., 0, 7.5, 4., 1.6, 9.9),
                          (5., 1., 0.5, 7., 3., 8., 9., 0., 7., 4., 1., 9.9)])
def test_periodic_init_types(parameters):
    """
    Tests that initializing a Periodic InteractionFunction with an n value (of
    any order) which is not an int, raises a TypeError
    """

    with pytest.raises(TypeError):
        Periodic(*parameters)

@pytest.mark.parametrize("parameters",
                         [(3., -1, 2.),
                          (5., 1, -30., 7., -3, 45.),
                          (9., 3, -40., 20., 4, -45., 60., -7, 9.),
                          (5., 1, 0.5, 7., 3, 8., 9., 0, 7.5, 4., -1, 9.9),
                          (5., -1, 0.5, 7., -3, 8., 9., -10, 7., 4., -1, 9.9)])
def test_periodic_init_values(parameters):
    """
    Tests that initializing a Periodic InteractionFunction with an n value (of
    any order) which is negative, raises a ValueError
    """

    with pytest.raises(ValueError):
        Periodic(*parameters)
