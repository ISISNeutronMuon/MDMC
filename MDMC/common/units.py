"""Module for all unit definitions and operations

This includes defining units used in MDMC, converting units, and subclassing
data strucures (float, NumPy array) so that they have a unit attribute.  This
style follows that of the Atomic Simulation Environment."""

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

    Parameters
    ----------
    string : str
        The unit, which can contain / to specify divisors and ^ to specify
        powers. It should not contain integers which are not used to specifying
        powers (e.g. '1 / Ang').  It cannot not contain negative powers (e.g.
        Ang ^ -1).
    components : defaultdict(list), optional
        Sets the components attribute (see Attributes).  Default is None.

    Examples
    --------
    Base units can be set::

    >>> time_unit = Unit('s')

    Compound units can be set with spaces separating base units which are
    multiplied::

    >>> charge_unit = Unit('A s')

    Compound units can be set with / separating base units which are divided::

    >>> velocity_unit = Unit('m / s')

    Units raised to a positive power can be set with ^::

    >>> volume_unit = Unit('Ang ^ 3')

    Compound units can be set with a combination of these operands::

    >>> force_unit = Unit('kg m / s ^ 2')

    To set an inverse unit, the power operation must be applied to a Unit::

    >>> frequency = Unit('s') ** -1

    Attributes
    ----------
    components : defaultdict(list)
        Contains the components of the unit, separated into two lists (numerator
        and denominator) depending on which side of the fraction each component
        is on.  If the Unit is a base unit i.e. initialized using Unit(), then
        the components only has a numerator and this is the Unit's string.
        If it a combined unit (created by either __mul__, __div__ or __pow__)
        then the units which combined to form it make up the components.
    """

    def __new__(cls, string, components=None):

        if string is None:
            return None
        unit = super(Unit, cls).__new__(cls, string)
        if not components:
            components = defaultdict(list)
            # String is compound if it contains either ' ', '/' or '^' (e.g.
            # 'Ang^2')
            if any(x in string for x in [' ', '/', '^']):
                num, denom = unit._parse_unit_string(string)
                components['numerator'] = num
                components['denominator'] = denom
            else:
                components['numerator'].append(unit)
                components['denominator'] = []
        unit.components = components
        return unit

    def __mul__(self, other):

        """
        Multiplies the unit by another unit

        Parameter
        ---------
        other : unit
            The unit object to multiply by

        Returns
        -------
        unit
            A compound unit
        """

        try:
            components = self._calculate_components(other, 'mul')
        except AttributeError:
            raise TypeError('A Unit can only be multipled by another Unit')
        return self.__class__(self._calculate_string(components), components)

    def __div__(self, other):

        """
        Divides the unit by another unit

        Parameter
        ---------
        other : unit
            The unit object to divide by

        Returns
        -------
        unit
            A compound unit
        """

        try:
            components = self._calculate_components(other, 'div')
        except AttributeError:
            raise TypeError('A Unit can only be divided by another Unit')
        return self.__class__(self._calculate_string(components), components)

    def __pow__(self, other):

        """
        Performs the power operation on the unit

        Parameter
        ---------
        other : numeric (inherits from numbers.Number)
            The number the unit is raised to the power of

        Returns
        -------
        unit
            A compound unit
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

        """
        Get whether the unit is a base or compound unit

        Returns
        -------
        bool
            If True, unit is a base unit (only has a single element in the
            components numerator list)
        """

        if (not self.components['denominator']
                and self.components['numerator'] == [self]):
            return True
        return False

    def _calculate_components(self, other, op):

        """
        Calculates the components for a new Unit generated from an operation

        These components are separated into whether they are in the numerator or
        the denominator of the new Unit.

        Parameters
        ----------
        other : unit
            the unit object which is operating on this unit object (i.e. self)
        op : str
            an operation, either 'mul', 'div', or 'pow'

        Returns
        -------
        defaultdict(list)
            contains the numerator and denominator of the new unit generated
            from the operation
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

        Parameters
        ----------
        components : defaultdict(list)
            contains the numerator and denominator of the new unit

        Returns
        -------
        str
            the string representing the new unit
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

    def _parse_unit_string(self, unit_string):

        """
        Converts a unit string into Unit objects

        Parameters
        ----------
        unit_string : str
            a string representing a unit

        Returns
        -------
        tuple
            A tuple of (numerator, denominator), where each is a list of unit
            objects for all of the base units.

        Example
        -------
        Parse 'e mol / K ^ 2'::

            >>> parse_unit_string('e mol / K ^ 2')
            ([Unit('e'), Unit('e'), Unit('mol')], [Unit('K'), Unit('K')])
        """

        def parse_powers(string):

            """
            Parameters
            ----------
            string : str
                a compound unit string containing zero or more powers
                (with powers specified by '^') but no denominators (i.e. '/'),
                such as 'Ang', 'Ang mol', 'Ang ^ 2 mol kJ^2'.

            Returns
            -------
            list
                contains all base units

            Example
            -------
            Parse 'Ang ^ 2 mol kJ^2'::
                >>> parse_powers('Ang ^ 2 mol kJ^2')
                [Unit('Ang'), Unit('Ang'), Unit('mol)', Unit('kJ'), Unit('kJ')]
            """

            if '^' in string:
                # Joining with ' ' before stripping out spaces means that
                # 'Ang ^ 2' and 'Ang^2' are equivalent
                string = ' '.join(string.split('^'))
            splt_space = string.split(' ')
            # Strip out spaces
            strip = filter(lambda x: x != '', splt_space)
            parsed = [Unit(strip[0])]
            # For all elements apart from the first, determine it element is a
            # digit. If so, append n-1 copies of the previous unit, where n is
            # the integer value of the element. If not, append a Unit object
            # initialized from the element (which should be a string specifying)
            # a unit
            for i in range(1, len(strip)):
                element = strip[i]
                if element.isdigit():
                    for _ in range(int(element) - 1):
                        parsed.append(Unit(strip[i-1]))
                else:
                    parsed.append(Unit(element))
            return parsed

        # Start by splitting the compound unit into a numerator and denominator
        if '/' in unit_string:
            num_string, denom_string = unit_string.split('/')
            denom = parse_powers(denom_string)
        else:
            num_string = unit_string
            denom = []
        num = parse_powers(num_string)

        return num, denom

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

    Parameters
    ----------
    codata_version : str
        the CODATA version to be used

    Returns
    -------
    dict
        contains (unit: conversion factor) pairs
    """

    # SYSTEM units are defined to 1.0
    units = {unit:1.0 for unit in SYSTEM.values()}

    # CODATA version
    codata = CODATA[codata_version]

    # Length
    # 1 m = 1e10 Ang
    units['m'] = units['Ang'] / 1e10
    # 1 nm = 1e1 Ang
    units['nm'] = units['Ang'] / 1e1

    # Time
    # 1 ns = 1e6 fs
    units['ns'] = units['fs'] / 1e6
    # 1 ns = 1e3 fs
    units['ps'] = units['fs'] / 1e3

    # Mass
    # 1 kg = (1000 * N_av) amu = (1/u) amu
    units['kg'] = units['amu'] * codata['_amu']
    # 1 g = N_av amu = (1/1000u) amu =
    units['g'] = units['kg'] * 1000.
    # 1 g mol^-1 = 1 amu by definition
    units['gmol'] = units['amu']

    # Energy
    # 1 J = (1/1000) kJ
    units['J'] = units['kJ'] * 1000.
    # 1 kcal = 4.184 kJ
    units['kcal'] = units['kJ'] / 4.184

    # Force
    # 1 kcal Ang^-1 mol^-1 = 4.184 kJ Ang^-1 mol^-1
    units['kcal / Ang mol'] = units['kJ / Ang mol'] / 4.184

    # Pressure
    # 1 atm = 101325 Pa
    units['atm'] = units['Pa'] / 101325.
    # 1 bar = 1e5 Pa
    units['bar'] = units['Pa'] / 1e5

    # Angle
    # 1 rad = (180 / pi) deg 
    units['rad'] = units['deg'] / (180. / np.pi)

    return units


class UnitFloat(float):

    """
    Subclasses float so that it contains a unit attribute

    Unit attribute is returned when __repr__ or __str__ are called.

    Parameters
    ----------
    value : float
        the value of the UnitFloat.
    unit : unit, str
        a unit or a string representing the unit.

    Note
    ----
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

        """
        Get or set the unit

        Either a str or a unit can be passed to the setter.

        Returns
        -------
        unit
            The unit object equivalent to the passed unit parameter.
        """

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
    Subclasses ndarray so that it contains a unit attribute

    Unit attribute is returned when __repr__ or __str__ are called

    Parameters
    ----------
    shape : tuple of ints
        Shape of created array.
    unit : unit, str
        a unit or a string representing the unit.
    dtype : data-type, optional
        Any object that can be interpreted as a NumPy data type.
    buffer : object exposing NumPy buffer interface, optional
        Used to fill the array with data.
    offset : int, optional
        Offset of array data in buffer.
    strides : tuple of ints, optional
        Strides of data in memory.
    order : str, optional
        Either 'C' for row-major or 'F' for column-major. Default is 'C'.
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

        """
        Get or set the unit

        Either a str or a unit can be passed to the setter.

        Returns
        -------
        unit
            The unit object equivalent to the passed unit parameter.
        """

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

    Parameters
    ----------
    object : None or array_like
        An object derived from (collections.Sequence). If None, then None is
        returned.
    unit : unit, str
        a unit or a string representing the unit.
    dtype : data-type, optional
        Any object that can be interpreted as a NumPy data type.

    Returns
    -------
    UnitArray
        A UnitArray object satisfying the specified requirements.
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
