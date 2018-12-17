"""Tests units assigned to properties

Tests properties belonging to the following classes: StructuralUnit, Atom,
Molecule, BoundingBox, MMTKEngine, Parameter, LAMPSQW, netCDF, xml_SQw, SQw

AUTHOR :    Thomas Farmer        START DATE :    17/12/2018, 13:12:24"""

from copy import deepcopy

import numpy as np
import pytest

from tests.MD.test_simulation_mmtk import water_MMTK_NVE

from MDMC.common import units
from MDMC.MD.engine_facades.mmtk import MMTKEngine
from MDMC.MD.interaction_functions import Parameter
from MDMC.MD.structural_units import Atom, Molecule, BoundingBox, Bond
from MDMC.MD.simulation import Universe, Shape


"""
Parameter:
value
constraints

Readers (LAMPSQW, netCDF, xml_SQw):
E
Q

SQw:
E
Q
SQw
SQw_err
time
"""

FLOAT = 50.0
LIST = [FLOAT, FLOAT, FLOAT]
SPCE_CHARGE = 0.4238
ERROR_MESSAGE = 'One or more {} properties has invalid values or units'


@pytest.fixture
def atom():

    return Atom('H', position=LIST, velocity=LIST, mass=FLOAT)


@pytest.fixture
def universe():

    return Universe(dimensions=LIST, shape=Shape.orthorhombic)


@pytest.fixture
def molecule(atom):

    atom2 = deepcopy(atom)
    return Molecule(position=LIST, atoms=[atom, atom2], name='Test',
                    interactions=[Bond(atom, atom2)])


def test_Atom_units(atom, universe):

    """
    Test the units of:

    position
    velocity
    mass
    charge
    """

    atom2 = deepcopy(atom)

    try:
        check_property(atom.position, LIST, units.LENGTH, units.unit_array)
        check_property(atom.velocity, LIST, units.LENGTH / units.TIME,
                       units.unit_array)
        check_property(atom.mass, FLOAT, units.MASS, units.UnitFloat)
    except AssertionError:
        raise AssertionError(ERROR_MESSAGE.format('Atom'))

    universe.add_structural_unit(atom)
    universe.add_force_field('SPCE')
    try:
        check_property(atom.charge, SPCE_CHARGE, units.CHARGE, units.UnitFloat)
    except AssertionError:
        raise AssertionError(ERROR_MESSAGE.format('Atom'))


def test_Molecule_units(molecule):

    """
    Test the units of:

    position
    """

    try:
        check_property(molecule.position, LIST, units.LENGTH, units.unit_array)
    except AssertionError:
        raise AssertionError(ERROR_MESSAGE.format('Molecule'))


def test_BoundingBox_units(molecule):

    """
    Test the units of:

    min
    max
    """

    box = BoundingBox(molecule.position, molecule.atom_list)

    try:
        check_property(box.min, LIST, units.LENGTH, units.unit_array)
        check_property(box.max, LIST, units.LENGTH, units.unit_array)
    except AssertionError:
        raise AssertionError(ERROR_MESSAGE.format('BoundingBox'))


def test_MMTKEngine_units():

    """
    Test the units of:

    temperature
    temperature_variation
    time_step
    pressure
    """

    mmtk_engine = MMTKEngine()
    mmtk_engine.temperature = FLOAT
    mmtk_engine.temperature_variation = FLOAT / 5.
    mmtk_engine.time_step = FLOAT
    mmtk_engine.pressure = FLOAT

    try:
        check_property(mmtk_engine.temperature, FLOAT, units.TEMPERATURE,
                       units.UnitFloat)
        check_property(mmtk_engine.temperature_variation, FLOAT  /5.,
                       units.TEMPERATURE, units.UnitFloat)
        check_property(mmtk_engine.time_step, FLOAT, units.TIME,
                       units.UnitFloat)
        check_property(mmtk_engine.pressure, FLOAT, units.PRESSURE,
                       units.UnitFloat)
    except AssertionError:
        raise AssertionError(ERROR_MESSAGE.format('MMTKEngine'))


def test_Parameter_units():

    """
    Test the units of:

    value
    constraints
    """

    # Values passed to Parameter must have units
    UNIT = units.ENERGY / units.AMOUNT * units.ANGLE**2
    CONSTRAINTS = [FLOAT-1, FLOAT+1]
    param1 = Parameter(units.UnitFloat(FLOAT, UNIT), 'test')
    param2 = Parameter(units.UnitFloat(FLOAT, UNIT), 'test',
                       constraints=CONSTRAINTS)

    def check_Parameter_properties(param, value, constraints):
        try:
            check_property(param.value, value, UNIT, units.UnitFloat)
            check_property(param.constraints, constraints, UNIT,
                           units.unit_array)
        except AssertionError:
            raise AssertionError(ERROR_MESSAGE.format('Parameter'))

    check_Parameter_properties(param1, FLOAT, None)
    check_Parameter_properties(param2, FLOAT, CONSTRAINTS)

    param1.constraints = CONSTRAINTS
    check_Parameter_properties(param1, FLOAT, CONSTRAINTS)


def check_property(prop, value, unit, cls):

    """
    Checks if the property is a float or array with the correct representation

    Arguments:
    prop - the property to be checked
    value - the expected float or array, or None
    unit - the expected string
    cls - either UnitFloat or unit_array (which is actually a helper function)
    """

    if value is None:
        expected = None
    else:
        expected = cls(value, unit)
    try:
        assert prop == expected
    except ValueError:
        assert all(prop == expected)
