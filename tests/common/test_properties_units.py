"""Tests units assigned to properties

Tests properties belonging to the following classes: Structure, Atom,
Molecule, BoundingBox, Parameter, LAMPSQW, MantidSQw, netCDF, xml_SQw, SQw"""

import numpy as np
import pytest

from MDMC.common import units
from MDMC.MD.interaction_functions import Parameter
from MDMC.MD.structures import Atom, Molecule, BoundingBox
from MDMC.MD.interactions import Bond
from MDMC.MD.simulation import Universe
from MDMC.readers.observables.obs_reader_factory import ObservableReaderFactory
from MDMC.trajectory_analysis.observables.sqw import SQw
from tests.test_data import data

FLOAT = 50.0
LIST = [FLOAT, FLOAT, FLOAT]
SPCE_CHARGE = 0.4238
ERROR_MESSAGE = 'One or more {} properties has invalid values or units'


@pytest.fixture
def atom():
    return Atom('H', position=LIST, velocity=LIST, mass=FLOAT)


@pytest.fixture
def universe():
    return Universe(dimensions=LIST)


@pytest.fixture
def molecule(atom):
    atom2 = atom.copy(atom.position)
    return Molecule(position=LIST, atoms=[atom, atom2], name='Test',
                    interactions=[Bond(atom, atom2)])


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

    box = BoundingBox(molecule.atoms)

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
    UNIT = units.ENERGY / units.ANGLE ** 2
    CONSTRAINTS = [FLOAT - 1, FLOAT + 1]
    parameter1 = Parameter(units.UnitFloat(FLOAT, UNIT), 'test')
    parameter2 = Parameter(units.UnitFloat(FLOAT, UNIT), 'test',
                           constraints=CONSTRAINTS)

    def check_Parameter_properties(parameter, value, constraints):
        try:
            check_property(parameter.value, value, UNIT, units.UnitFloat)
            check_property(parameter.constraints, constraints, UNIT,
                           units.unit_array)
        except AssertionError:
            raise AssertionError(ERROR_MESSAGE.format('Parameter'))

    check_Parameter_properties(parameter1, FLOAT, None)
    check_Parameter_properties(parameter2, FLOAT, CONSTRAINTS)

    parameter1.constraints = CONSTRAINTS
    check_Parameter_properties(parameter1, FLOAT, CONSTRAINTS)


Q_UNIT = units.LENGTH ** -1
E_UNIT = units.ENERGY_TRANSFER
READERS_TEST_INFO = [('LAMPSQw', 'LAMPSQw', [{'name': 'Q', 'value': LIST, 'unit': Q_UNIT},
                                             {'name': 'E', 'value': LIST, 'unit': E_UNIT}]
                      ),
                     ('MantidSQw', 'MantidSQw_one_file', [{'name': 'Q', 'value': LIST, 'unit': Q_UNIT},
                                                          {'name': 'E', 'value': LIST, 'unit': E_UNIT}]
                      ),
                     ('MantidSQw', 'MantidSQw_two_files', [{'name': 'Q', 'value': LIST, 'unit': Q_UNIT},
                                                           {'name': 'E', 'value': LIST, 'unit': E_UNIT}]
                      ),
                     ('xml_SQw', 'xml_SQw', [{'name': 'Q', 'value': LIST, 'unit': Q_UNIT},
                                             {'name': 'E', 'value': LIST, 'unit': E_UNIT}]
                      )]


@pytest.fixture(params=READERS_TEST_INFO)
def reader_info(request):
    """
    Parameterized reader instantiation

    Returns:
    A dictionary of the name of the reader, a reader with the properties set,
    and the names, values and units of the properties for testing
    """

    reader = ObservableReaderFactory.create(request.param[0],
                                            data.READER_DATA[request.param[1]])
    for prop in request.param[2]:
        setattr(reader, prop['name'], prop['value'])

    return {'reader_name': request.param[0],
            'reader': reader,
            'properties': request.param[2]}


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
    """

    sqw = SQw()
    sqw.independent_variables = {'E': LIST, 'Q': LIST}
    sqw._dependent_variables = {'SQw': [LIST, LIST, LIST]}
    sqw._errors = {'SQw': [LIST, LIST, LIST]}

    try:
        check_property(sqw.E, LIST, units.ENERGY_TRANSFER, units.unit_array)
        check_property(sqw.Q, LIST, units.LENGTH ** -1, units.unit_array)
        check_property(sqw.SQw, [LIST, LIST, LIST], units.ARBITRARY,
                       units.unit_array)
        check_property(sqw.SQw_err, [LIST, LIST, LIST], units.ARBITRARY,
                       units.unit_array)
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
