"""Tests units assigned to properties

Tests properties belonging to the following classes: StructuralUnit, Atom,
Molecule, BoundingBox, Parameter, LAMPSQW, netCDF, xml_SQw, SQw

AUTHOR :    Thomas Farmer        START DATE :    17/12/2018, 13:12:24"""

import numpy as np
import pytest


from MDMC.common import units
from MDMC.MD.interaction_functions import Parameter
from MDMC.MD.structural_units import Atom, Molecule, BoundingBox, Bond, \
                                     Coulombic
from MDMC.MD.simulation import Universe, Shape
from MDMC.readers.observables.obs_reader_factory import ObservableReaderFactory
from MDMC.trajectory_analysis.observables.SQw import SQw


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

    atom2 = atom.copy(atom.position)
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

    atom_coulombic = Coulombic(atoms=atom)

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

    box = BoundingBox(molecule.atom_list)

    try:
        check_property(box.min, LIST, units.LENGTH, units.unit_array)
        check_property(box.max, LIST, units.LENGTH, units.unit_array)
    except AssertionError:
        raise AssertionError(ERROR_MESSAGE.format('BoundingBox'))


def test_Parameter_units():

    """
    Test the units of:

    value
    constraints - initialized to None, initialized to FLOAT, and set to FLOAT
    """

    # Values passed to Parameter must have units
    UNIT = units.ENERGY / units.ANGLE**2
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


Q_UNIT = units.LENGTH ** -1
E_UNIT = units.ENERGY_TRANSFER
READERS_TEST_INFO = [('LAMPSQw', [{'name':'Q', 'value':LIST, 'unit':Q_UNIT},
                                  {'name':'E', 'value':LIST, 'unit':E_UNIT}]
                     ),
                     ('netCDF', [{'name':'Q', 'value':LIST, 'unit':Q_UNIT},
                                 {'name':'E', 'value':LIST, 'unit':E_UNIT}]
                     ),
                     ('xml_SQw', [{'name':'Q', 'value':LIST, 'unit':Q_UNIT},
                                  {'name':'E', 'value':LIST, 'unit':E_UNIT}]
                     )]

@pytest.fixture(params=READERS_TEST_INFO)
def reader_info(request):

    """
    Parameterized reader instantiation

    Returns:
    A dictionary of the name of the reader, a reader with the properties set,
    and the names, values and units of the properties for testing
    """

    reader = ObservableReaderFactory.create_reader(request.param[0])
    for prop in request.param[1]:
        setattr(reader, prop['name'], prop['value'])

    return {'reader_name':request.param[0],
            'reader':reader,
            'properties':request.param[1]}


def test_Reader_units(reader_info):

    """
    Test the units for all Readers:
    """
    try:
        for prop in reader_info['properties']:
            check_property(getattr(reader_info['reader'], prop['name']),
                           prop['value'],
                           prop['unit'],
                           units.UnitFloat if isinstance(prop['unit'], float)
                           else units.unit_array)
    except AssertionError:
        raise AssertionError(ERROR_MESSAGE.format(reader_info['reader_name']))


def test_SQw_units():

    """
    Test the units of:

    E
    Q
    SQw
    SQw_err
    time
    t_res
    """

    sqw = SQw()
    sqw.independent_variables = {'E':LIST,'Q':LIST}
    sqw._dependent_variables = {'SQw':[LIST, LIST, LIST]}
    sqw._errors = {'SQw':[LIST, LIST, LIST]}
    sqw.t = LIST
    sqw.t_res = FLOAT

    try:
        check_property(sqw.E, LIST, units.ENERGY_TRANSFER, units.unit_array)
        check_property(sqw.Q, LIST, units.LENGTH ** -1, units.unit_array)
        check_property(sqw.SQw, [LIST, LIST, LIST], units.ARBITRARY,
                       units.unit_array)
        check_property(sqw.SQw_err, [LIST, LIST, LIST], units.ARBITRARY,
                       units.unit_array)
        check_property(sqw.t, LIST, units.TIME, units.unit_array)
        check_property(sqw.t_res, FLOAT, units.TIME, units.UnitFloat)
    except AssertionError:
        raise AssertionError(ERROR_MESSAGE.format('SQw'))


def check_property(prop, value, unit, cls):

    """
    Checks if the property is a float or array with the correct representation
    and value

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
    assert repr(prop) == repr(expected)
    try:
        assert np.all(prop == expected)
    except TypeError:
        assert prop == expected
