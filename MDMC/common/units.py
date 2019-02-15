"""Module for all unit definitions and operations

This includes defining units used in MDMC, converting units, and subclassing
data strucures (float, NumPy array) so that they have a unit attribute.  This
style follows that of the Atomic Simulation Environment.

AUTHOR :    Thomas Farmer        START DATE :    12/12/2018, 11:06:51"""

from collections import Counter, defaultdict
from copy import deepcopy
from numbers import Number

import numpy as np


CODATA_VERSION = '2014'


CODATA = {

    # CODATA 2014 taken from ASE.units, originally from:
    # http://arxiv.org/pdf/1507.07956.pdf
    '2014': {'_c': 299792458.,
             '_mu0': 4.0e-7 * np.pi,
             '_Grav': 6.67408e-11,
             '_hplanck': 6.626070040e-34,
             '_e': 1.6021766208e-19,
             '_me': 9.10938356e-31,
             '_mp': 1.672621898e-27,
             '_Nav': 6.022140857e23,
             '_k': 1.38064852e-23,
             '_amu': 1.660539040e-27}
}


class Unit(str):

    """
    A class for defining unit strings

    It possesses additional * and / operands so that combined units can be
    returned.

    NON-INTEGER POWER OPERATIONS ARE CURRENTLY NOT IMPEMENTED

    Attributes:
    components - a defaultdict(list) containing the components of the unit,
    separated into two lists (numerator and denominator) depending on which side
    of the fraction each component is on.  If the Unit is a base unit i.e.
    initialized using Unit(), then the components only has a numerator and this
    is the Unit's string.  If it a combined unit (created by either __mul__,
    __div__ or __pow__) then the units which combined to form it make up the
    components.
    """

    def __new__(cls, string, components=None):

        """
        Arguments:
        string - a string specifying the unit
        components - a defaultdict(list) specifying the numerator and
        denominator components of the Unit
        """

        unit = super(Unit, cls).__new__(cls, string)
        if not components:
            components = defaultdict(list)
            components['numerator'].append(unit)
            components['denominator'] = []
        unit.components = components
        return unit

    def __mul__(self, other):

        """
        Multiplies the unit by unit

        Arguments:
        other - a unit
        """

        try:
            components = self._calculate_components(other, 'mul')
        except AttributeError:
            raise TypeError('A Unit can only be multipled by another Unit')
        return self.__class__(self._calculate_string(components), components)

    def __div__(self, other):

        """
        Divides the unit by another unit

        Arguments:
        other - a unit
        """

        try:
            components = self._calculate_components(other, 'div')
        except AttributeError:
            raise TypeError('A Unit can only be divided by another Unit')
        return self.__class__(self._calculate_string(components), components)

    def __pow__(self, other):

        """
        Performs the power operation on the unit

        Arguments:
        other - a numeric type (inherits from numbers.Number)
        """

        if not isinstance(other, Number):
            try:
                other = float(other)
            except (TypeError, ValueError) as _:
                raise TypeError('Only numeric types can be used as a power for'
                                ' Units')

        components = self._calculate_components(other, 'pow')
        return self.__class__(self._calculate_string(components), components)

    @property
    def base(self):

        if (not self.components['denominator']
                and self.components['numerator'] == [self]):
            return True
        return False

    def _calculate_components(self, other, op):

        """
        Calculates the components for a new Unit generated from an operation

        These components are separated into whether they are in the numerator or
        the denominator of the new Unit

        Arguments:
        other - another Unit object
        op - a string specifying an operation

        Returns:
        A defaultdict(list) containing the numerator and denominator of the
        new Unit
        """

        # Creating another defaultdict and then populating it by deepcopying
        # every unit in the numerator and denominator avoids issues with
        # multiple component dictionaries referencing the same object - this
        # previously led to units which were base units being transformed into
        # combined units as the lists in their components dictionary were
        # modified
        components = defaultdict(list)
        for k, lst in self.components.items():
            components[k] = [deepcopy(unit) for unit in lst]
        if op == 'mul':
            components['numerator'] += other.components['numerator']
            components['denominator'] += other.components['denominator']
        if op == 'div':
            components['numerator'] += other.components['denominator']
            components['denominator'] += other.components['numerator']
        if op == 'pow':
            # Ensure other is an integer
            other = int(other)
            if other >= 1:
                components['numerator'] *= other
                components['denominator'] *= other
            else:
                numerator = components['numerator']
                components['numerator'] = components['denominator'] * abs(other)
                components['denominator'] = numerator * abs(other)

        return components

    def _calculate_string(self, components):

        """
        Calculates the string for a new Unit generated from an operation

        Arguments:
        components - a defaultdict(list) containing the numerator and
        denominator of the new Unit

        Returns:
        A string specifying the new Unit
        """

        def _calculate_expr_string(expr):

            """
            Calculates the string from a list of components

            Counter is used to determined the number of occurences of each unit
            string and then create power notation if there is more than one
            occurence.
            """

            component_powers = Counter(expr)
            # List used rather than string so that sorting can be implemented
            component_list = []
            for comp, power in component_powers.items():
                if power is 1:
                    component_list.append(comp)
                else:
                    component_list.append(comp + ' ^ ' + str(power))
            return ' '.join(component_list)

        numerator = _calculate_expr_string(components['numerator'])
        denominator = _calculate_expr_string(components['denominator'])

        # Different string styles for the three cases of just numerator, just
        # denominator, and both
        if not components['numerator']:
            return '1 / ' + denominator
        else:
            if not components['denominator']:
                return numerator
            else:
                return numerator + ' / ' + denominator


# Define the unit system used in MDMC
SYSTEM = {
    'LENGTH':Unit('Ang'),
    'TIME':Unit('fs'),
    'MASS':Unit('amu'),
    'CHARGE':Unit('e'),
    'ANGLE':Unit('deg'),
    'TEMPERATURE':Unit('K'),
    'AMOUNT':Unit('mol'),
    'ENERGY':Unit('kJ'),
    'FORCE':Unit('kJ') / (Unit('mol') * Unit('Ang')),
    'PRESSURE':Unit('Pa'),
    'ENERGY_TRANSFER':Unit('meV'),
    'ARBITRARY':Unit('arb')
}


def create_units(codata_version):

    """
    Creates a dictionary of units based on the CODATA version.

    Arguments:
    codata_version - str specifying the CODATA version to be used

    Returns:
    dictionary containing (unit, conversion factor) pairs
    """

    # SYSTEM units are defined to 1.0
    units = {unit:1.0 for unit in SYSTEM.values()}

    # CODATA version
    codata = CODATA[codata_version]

    # Length
    units['m'] = 1e10
    units['nm'] = 10.

    # Time
    units['ns'] = 1e6
    units['ps'] = 1e3

    # Mass
    units['kg'] = 1. / codata['_amu']
    units['g'] = units['kg'] * 1000.

    # Energy
    units['J'] = units['kJ'] * 1000.
    units['kcal'] = units['kJ'] / 4.184

    # Force
    units['kcal / Ang mol'] = units['kJ / Ang mol'] / 4.184

    # Pressure
    units['atm'] = units['Pa'] / 101325.
    units['bar'] = units['Pa'] / 1e5

    # Angle
    units['rad'] = units['deg'] * np.pi / 180.

    return units


class UnitFloat(float):

    """
    Subclasses float so that it contains a unit attribute which is returned when
    __repr__ or __str__ are called

    Attributes:
    unit - a string or Unit which specifies the unit

    NB:
    As both __repr__ and __deepcopy__ rely on the float being real, this class
    is not compatible with complex numbers.  This should be immaterial as no
    quantity which possesses units is complex.
    """

    def __new__(cls, value, unit):

        if value is None:
            return None
        return float.__new__(cls, value)

    def __init__(self, value, unit):

        float.__init__(value)
        self.unit = unit

    @property
    def unit(self):

        return self._unit

    @unit.setter
    def unit(self, value):

        if not (isinstance(value, str) or value is None):
            raise TypeError('unit must be a string')
        self._unit = Unit(value)

    def __deepcopy__(self, memo):

        """
        Copies the UnitFloat and all attributes

        This method is required because otherwise the float.__deepcopy__ is
        used, which attempts to create a new UnitFloat class using only 2
        argument i.e. the value.  UnitFloat.__new__ takes exactly 3 arguments.

        It simply creates a new UnitFloat and sets all of its attributes to
        deepcopies of the current attributes (where possible), along with
        updating the memo.
        """

        cls = self.__class__
        unit_float = cls.__new__(cls, self.real, self.unit)
        memo[id(self)] = unit_float
        for k, v in self.__dict__.items():
            setattr(unit_float, k, deepcopy(v, memo))
        return unit_float

    def __repr__(self):

        return repr(self.real) + ' ' + self.unit

    def __str__(self):

        return  self.__repr__()


class UnitNDArray(np.ndarray):

    """
    Subclasses ndarray so that it contains a unit attribute which is returned
    when __repr__ or __str__ are called

    Attributes:
    unit - a string or Unit which specifies the unit
    """

    def __new__(cls, shape, unit, dtype=float, buffer=None, offset=0,
                strides=None, order=None):
        obj = super(UnitNDArray, cls).__new__(cls, shape, dtype,
                                              buffer, offset, strides,
                                              order)
        obj.unit = unit
        return obj

    def __array_finalize__(self, obj):
        self.unit = getattr(obj, 'unit', None)

    @property
    def unit(self):

        return self._unit

    @unit.setter
    def unit(self, value):

        if not (isinstance(value, str) or value is None):
            raise TypeError('unit must be a string')
        self._unit = Unit(value)

    def __repr__(self):
        try:
            return super(UnitNDArray, self).__repr__() + ' ' + self.unit
        except TypeError:
            return super(UnitNDArray, self).__repr__()

    def __str__(self):

        return  self.__repr__()


def unit_array(obj, unit, dtype=None):

    """
    Helper function for creating a UnitNDArray from an array or any nested
    sequence

    This mimics the manner in which numpy creates arrays (although is in python
    not C), except several arguments are excluded.

    Also, unlike np.array(None), passing obj=None to unit_array results in None
    being returned. This allows classes to have properties with units which can
    be either have a value or be undefined.

    Arguments:
    object - an array or array-like object (e.g. any object derived from
    collections.Sequence). If None, then None is returned.
    unit - a string or Unit which specifies the unit of the array
    dtype - the desired data-type for the array
    """

    if obj is None:
        return None

    if not isinstance(unit, str):
        raise TypeError('unit must be a string')

    # Significantly faster to create np.array and view it than to loop
    if not isinstance(obj, np.ndarray):
        obj = np.array(obj, dtype=dtype)

    unit_arr = obj.view(UnitNDArray)
    unit_arr.unit = unit
    return unit_arr


# Update the module scope to include the SYSTEM and units keys
globals().update(SYSTEM)
globals().update(create_units(CODATA_VERSION))
