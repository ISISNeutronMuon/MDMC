"""
Module for all unit definitions and operations.

This includes defining units used in MDMC, converting units, and subclassing
data strucures (`float`, ``array``) so that they have a ``unit``
attribute.  This style follows that of the Atomic Simulation Environment.

As members of units.py are set dynamically, pylintrc excludes member checking
for units.py using the generated-members keyword. Care must be taken to ensure
members exist when importing from units.py, as these will not be linted.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from numbers import Number
from typing import Union

import numpy as np


# pylint: disable=no-member
# as it raises a false positive for components

#: Version of CODATA used.
CODATA_VERSION = '2014'


#: CODATA 2014 taken from ASE.units, originally from:
#: http://arxiv.org/pdf/1507.07956.pdf
CODATA = {

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
    A class for defining unit strings.

    It possesses additional ``*`` and ``/`` operands so that combined units can
    be returned.

    .. note:: NON-INTEGER POWER OPERATIONS ARE CURRENTLY NOT IMPEMENTED

    Parameters
    ----------
    string : str
        The unit, which can contain ``/`` to specify divisors and ``^`` to
        specify powers. It can contain `int` which are used to specify order of
        magnitude (e.g. ``10^6 Pa``).  It can also contain negative powers, but
        there must not be a space between the negative sign and the number
        (e.g. ``Ang ^ -1`` NOT ``Ang ^ - 1``). Brackets and parentheses are not
        supported, and any of the characters ``[]()`` will be ignored.
    components : defaultdict(list), optional
        Sets the ``components`` attribute (see Attributes).  Default is `None`.

    Attributes
    ----------
    components : `defaultdict(list)`
        Contains the ``components`` of the ``Unit``, separated into two `list`
        (``numerator`` and ``denominator``) depending on which side of the
        fraction each component is on.  If the ``Unit`` is a ``base`` unit i.e.
        initialized using ``Unit()``, then the ``components`` only has a
        ``numerator`` and this is the ``Unit`` string. If it a combined ``Unit``
        (created by either ``__mul__``, ``__div__`` or ``__pow__``) then the
        ``Unit`` objects which combined to form it make up the ``components``.
    conversion_factor : float
        The factor by which to multiply a value in order to express it in
        system units. For example ``Unit('ps').conversion_factor`` is
        ``1000.`` as the system units are femtoseconds.

    Examples
    --------
    Base units can be set::

    >>> time_unit = Unit('s')

    Compound units can be set with spaces separating ``base`` units which are
    multiplied::

    >>> charge_unit = Unit('A s')

    Compound units can be set with ``/`` separating ``base`` units which are
    divided::

    >>> velocity_unit = Unit('m / s')

    Units raised to a power can be set with ``^``::

    >>> volume_unit = Unit('Ang ^ 3')

    Compound units can be set with a combination of these operands::

    >>> force_unit = Unit('kg m / s ^ 2')

    To set an inverse unit, the power operation can be applied to a ``Unit``::

    >>> frequency = Unit('s') ** -1

    Or set with ``^`` within the string::

    >>> frequency = Unit('s ^ -1')

    Or with ``/`` within the string::

    >>> frequency = Unit('1 / s')

    Orders of magnitude can also be included::

    >>> pressure = Unit('10 ^ 6 Pa')
    """

    def __new__(cls, string: str, components: defaultdict[list] = None) -> Unit:

        if string is None:
            return None
        if isinstance(string, Unit):
            return string

        # Remove square brackets and parentheses as these are not supported
        unsupported_chars = ['[', ']', '(', ')']
        for char in unsupported_chars:
            string = string.replace(char, '')

        unit = super().__new__(cls, string)
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

    def __mul__(self, other: Unit) -> Unit: #type: ignore
        """
        Multiply the ``Unit`` by another ``Unit``.

        Parameters
        ----------
        other : Unit
            The ``Unit`` object to multiply by.

        Returns
        -------
        Unit
            A compound ``unit``.

        Raises
        ------
        TypeError
            If `other` is not a valid ``Unit``.
        """

        try:
            components = self._calculate_components(other, 'mul')
        except AttributeError as error:
            raise TypeError(
                'A Unit can only be multipled by another Unit') from error
        return self.__class__(self._calculate_string(components), components)

    def __truediv__(self, other: Unit) -> Unit:
        """
        Divide the ``Unit`` by another ``Unit``.

        Parameters
        ----------
        other : Unit
            The ``Unit`` object to divide by.

        Returns
        -------
        Unit
            A compound ``Unit``.

        Raises
        ------
        TypeError
            If `other` is not a valid ``Unit``.
        """

        try:
            components = self._calculate_components(other, 'div')
        except AttributeError as error:
            raise TypeError(
                'A Unit can only be divided by another Unit') from error
        return self.__class__(self._calculate_string(components), components)

    def __pow__(self, exponent: Number) -> Unit:
        """
        Perform the power operation on the ``Unit``.

        Parameters
        ----------
        exponent : numeric (inherits from numbers.Number)
            The number the ``Unit`` is raised to the power of.

        Returns
        -------
        Unit
            A compound ``Unit``.

        Raises
        ------
        TypeError
            If `exponent` a valid integer power.
        """

        if not isinstance(exponent, Number):
            try:
                exponent = float(exponent)
            except (TypeError, ValueError) as error:
                raise TypeError('Only numeric types can be used as a power for'
                                ' Units') from error

        components = self._calculate_components(exponent, 'pow')
        return self.__class__(self._calculate_string(components), components)

    @property
    def base(self) -> bool:
        """
        Get whether the ``Unit`` is a ``base`` or compound ``Unit``.

        Returns
        -------
        bool
            If `True`, ``Unit`` is a ``base`` ``Unit`` (only has a single
            element in the ``components`` ``numerator`` `list`).
        """

        if (not self.components['denominator']
                and self.components['numerator'] == [self]):
            return True
        return False

    @property
    def conversion_factor(self) -> float:
        """
        Get conversion factor for ``Unit``.

        Calculates the factor by which a value with this ``Unit`` should be
        multiplied in order to express it in system units. This takes into
        account any orders of magnitude and compound units. The
        ``conversion_factor`` for a ``Unit`` composed only of system units is
        therefore always 1.

        Returns
        -------
        float
            The conversion factor to system units for the relevant property.
        """

        factor = 1.
        factors_dict = create_units(CODATA_VERSION)[0]
        components = self.components
        numerator = components['numerator']
        denominator = components['denominator']

        try:
            for unit in numerator:
                try:
                    factor *= int(str(unit))
                except ValueError:
                    factor *= factors_dict[str(unit)]

            for unit in denominator:
                try:
                    factor /= int(str(unit))
                except ValueError:
                    factor /= factors_dict[str(unit)]

        except KeyError as error:
            raise KeyError(f'Unknown unit {str(unit)} provided, cannot convert to system'
                           ' units') from error

        return factor

    @property
    def physical_property(self) -> str:
        """
        The physical property (e.g. 'LENGTH', 'TIME', ...) that the unit measures.

        Note that compound units may not be supported, for the list of supported units see
        ``create_units``.

        Returns
        -------
        `str` or None
            The physical property.
        """

        properties_dict = create_units(CODATA_VERSION)[1]
        try:
            return properties_dict[str(self)]
        except KeyError as error:
            raise KeyError(f'Unknown unit {str(self)} provided, cannot determine the '
                           'physical property it measures ') from error

    def _calculate_components(self, other: Unit, op: str) -> defaultdict[list]:
        """
        Calculate the ``components`` for a new ``Unit`` generated from an operation.

        These ``components`` are separated into whether they are in the
        ``numerator`` or the ``denominator`` of the new ``Unit``.

        Parameters
        ----------
        other : unit
            The ``Unit`` which is operating on this ``Unit`` (i.e. ``self``).
        op : str
            An operation, either ``mul``, ``div``, or ``pow``.

        Returns
        -------
        `defaultdict(list)`
            Contains the ``numerator`` and ``denominator`` of the new ``Unit``
            generated from the operation.
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
                components['numerator'] = components['denominator'] * \
                    abs(other)
                components['denominator'] = numerator * abs(other)

        return components

    @staticmethod
    def _calculate_string(components: defaultdict(list)) -> str:
        """
        Calculate the `str` for a new ``Unit`` generated from an operation.

        Parameters
        ----------
        components : defaultdict(list)
            Contains the ``numerator`` and ``denominator`` of the new ``Unit``.

        Returns
        -------
        `str`
            The `str` representing the new ``Unit``.
        """

        def _calculate_expr_string(expr) -> str:
            """
            Calculate the `str` from a `list` of ``components``.

            ``Counter`` is used to determined the number of occurences of each
            ``Unit`` `str` and then create power notation if there is more than
            one occurence.

            Parameters
            ----------
            expr : str
                Add power to underlying expression.

            Returns
            -------
            str
                Unit expression as `str` with appropriate powers.
            """

            component_powers = Counter(expr)
            # List used rather than string so that sorting can be implemented
            component_list = []
            for comp, power in component_powers.items():
                if power == 1:
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
        if not components['denominator']:
            return numerator
        return numerator + ' / ' + denominator

    @staticmethod
    def _parse_unit_string(unit_string: str) -> 'tuple[list[Unit]]':
        """
        Convert a ``Unit`` `str` into ``Unit`` objects.

        Parameters
        ----------
        unit_string : str
            A `str` representing a ``Unit``.

        Returns
        -------
        tuple
            A `tuple` of (``numerator``, ``denominator``), where each is a
            `list` of ``Unit`` objects for all of the ``base`` ``Unit`` objects.

        Examples
        --------
        Parse ``e mol / K ^ 2``::

            >>> parse_unit_string('e mol / K ^ 2')
            ([Unit('e'), Unit('e'), Unit('mol')], [Unit('K'), Unit('K')])
        """

        def parse_powers(string):
            """
            Parse string to series of ``Unit`` s.

            Parameters
            ----------
            string : str
                A compound ``Unit`` `str` containing zero or more powers
                (with powers specified by ``^``) but no denominators (i.e.
                ``/``), such as ``Ang``, ``Ang mol``, ``Ang ^ 2 mol kJ^2``.

            Returns
            -------
            `tuple` of `list`
                Contains all ``base`` ``Unit``s that have postive powers and
                negative powers respectively.

            Examples
            --------
            Parse ``Ang ^ 2 mol kJ^-2``::
                >>> parse_powers('Ang ^ 2 mol kJ^2')
                [Unit('Ang'), Unit('Ang'), Unit('mol)'], [Unit('kJ'), Unit('kJ')]
            """

            if '^' in string:
                # Joining with ' ' before stripping out spaces means that
                # 'Ang ^ 2' and 'Ang^2' are equivalent
                string = ' '.join(string.split('^'))
            splt_space = string.split(' ')
            # Strip out spaces
            strip = list(filter(lambda x: x != '', splt_space))
            parsed = [Unit(strip[0])]
            inverse_parsed = []
            # For all elements apart from the first, determine if element is an
            # int. If so, and it is positive, append n-1 copies of the previous
            # unit, where n is the integer value of the element. If it is
            # negative, remove the previous element from the numerator and
            # instead add n copies to the denominator. If n is zero, simply
            # remove the element. If not an int, append a Unit object
            # initialized from the element (which should be a string specifying)
            # a unit
            for i in range(1, len(strip)):
                element = strip[i]
                try:
                    exponent = int(element)
                    if exponent > 0:
                        for _ in range(exponent - 1):
                            parsed.append(parsed[-1])
                    else:
                        base = parsed.pop()
                        for _ in range(- exponent):
                            inverse_parsed.append(base)
                except ValueError:
                    parsed.append(Unit(element))
            return parsed, inverse_parsed

        # Start by splitting the compound unit into a numerator and denominator
        if '/' in unit_string:
            num_string, denom_string = unit_string.split('/')
            denom, inverse_denom = parse_powers(denom_string)
        else:
            num_string = unit_string
            denom, inverse_denom = [], []
        num, inverse_num = parse_powers(num_string)

        # Combine the numerator with elements from the denominator that had
        # negative powers, and vice versa
        return num + inverse_denom, denom + inverse_num


# Define the unit system used in MDMC
SYSTEM = {
    'LENGTH': Unit('Ang'),
    'TIME': Unit('fs'),
    'MASS': Unit('amu'),
    'CHARGE': Unit('e'),
    'ANGLE': Unit('deg'),
    'TEMPERATURE': Unit('K'),
    'ENERGY': Unit('kJ') / Unit('mol'),
    'FORCE': Unit('kJ') / (Unit('Ang') * Unit('mol')),
    'PRESSURE': Unit('Pa'),
    'ENERGY_TRANSFER': Unit('meV'),
    'ARBITRARY': Unit('arb')
}


def create_units(codata_version: str) -> dict[Unit, float]:
    """
    Create a `dict` of ``Unit`` based on the CODATA version.

    Parameters
    ----------
    codata_version : str
        The CODATA version to be used.

    Returns
    -------
    dict
        Contains (``Unit``: conversion factor) pairs.
    """

    # SYSTEM units are defined to 1.0, and have the physical properties they
    # measure defined (e.g. {'Ang': 'LENGTH', ...})
    units = {unit: 1.0 for unit in SYSTEM.values()}
    unit_properties = {unit: property for property, unit in SYSTEM.items()}

    # CODATA version
    codata = CODATA[codata_version]

    # Length
    # 1 m = 1e10 Ang
    units['m'] = units['Ang'] * 1e10
    unit_properties['m'] = 'LENGTH'
    # 1 cm = 1e8 Ang
    units['cm'] = units['Ang'] * 1e8
    unit_properties['cm'] = 'LENGTH'
    # 1 nm = 1e1 Ang
    units['nm'] = units['Ang'] * 1e1
    unit_properties['nm'] = 'LENGTH'
    # 1 AA = 1 Ang
    units['AA'] = units['Ang']
    unit_properties['AA'] = 'LENGTH'

    # Area
    # 1 barn = 1e-28 m^2 = 1e-8 Ang^2
    units['barn'] = units['Ang'] * units['Ang'] * 1e8

    # Time
    # 1 s = 1e15 fs
    units['s'] = units['fs'] * 1e15
    unit_properties['s'] = 'TIME'
    # 1 ns = 1e6 fs
    units['ns'] = units['fs'] * 1e6
    unit_properties['ns'] = 'TIME'
    # 1 ps = 1e3 fs
    units['ps'] = units['fs'] * 1e3
    unit_properties['ps'] = 'TIME'

    # Mass
    # 1 kg = (1000 * N_av) amu = (1/u) amu
    units['kg'] = units['amu'] / codata['_amu']
    unit_properties['kg'] = 'MASS'
    # 1 g = N_av amu = (1/1000u) amu = (1/1000) kg
    units['g'] = units['kg'] / 1000.
    unit_properties['g'] = 'MASS'
    # 1 g mol^-1 = 1 amu by definition
    units['g / mol'] = units['amu']
    unit_properties['g / mol'] = 'MASS'

    # Energy
    # 1 kcal mol^-1 = 4.184 kJ mol^-1
    units['kcal / mol'] = units['kJ / mol'] * 4.184
    unit_properties['kcal / mol'] = 'ENERGY'
    # 1 kJ = 1 kJ mol^-1 * Nav
    units['kJ'] = units['kJ / mol'] * codata['_Nav']
    unit_properties['kJ'] = 'ENERGY'
    # 1 J = (1/1000) kJ
    units['J'] = units['kJ'] / 1000.
    unit_properties['J'] = 'ENERGY'
    # 1 kcal = 4.184 kJ
    units['kcal'] = units['kJ'] * 4.184
    unit_properties['kcal'] = 'ENERGY'

    # Energy transfer
    units['ueV'] = units['meV'] * 1000.
    unit_properties['ueV'] = 'ENERGY_TRANSFER'

    # Force
    # 1 kcal Ang^-1 mol^-1 = 4.184 kJ Ang^-1 mol^-1
    units['kcal / Ang mol'] = units['kJ / Ang mol'] * 4.184
    unit_properties['kcal / Ang mol'] = 'FORCE'

    # Pressure
    # 1 atm = 101325 Pa
    units['atm'] = units['Pa'] * 101325.
    unit_properties['atm'] = 'PRESSURE'
    # 1 bar = 1e5 Pa
    units['bar'] = units['Pa'] * 1e5
    unit_properties['bar'] = 'PRESSURE'

    # Angle
    # 1 rad = (180 / pi) deg
    units['rad'] = units['deg'] * (180. / np.pi)
    unit_properties['rad'] = 'ANGLE'

    # Amount
    units['mol'] = codata['_Nav']

    return units, unit_properties


class UnitFloat(float):
    """
    Subclass of `float` so that it contains a ``unit`` attribute.

    ``unit`` attribute is returned when ``__repr__`` or ``__str__`` are called.

    Parameters
    ----------
    value : float
        The value of the ``UnitFloat``.
    unit : Unit, str
        A ``Unit`` or a `str` representing the unit.

    Notes
    -----
    As both ``__repr__`` and ``__deepcopy__`` rely on the `float` being real,
    this class is not compatible with complex numbers.  This should be
    immaterial as no quantity which possesses units is complex.
    """

    def __new__(cls, value: float, unit: Union[Unit, str]):

        if value is None:
            return None
        return float.__new__(cls, value)

    def __init__(self, value: float, unit: Union[Unit, str]):

        float.__init__(value)
        self.unit = unit

    @property
    def unit(self) -> Unit:
        """
        Get or set the ``unit``.

        Either a `str` or a ``Unit`` can be passed to the setter.

        Returns
        -------
        Unit
            The ``Unit`` equivalent to the passed ``unit`` parameter.
        """

        return self._unit

    @unit.setter
    def unit(self, value: float) -> None:

        if not (isinstance(value, str) or value is None):
            raise TypeError('unit must be a string')
        self._unit = Unit(value)

    def __deepcopy__(self, memo: dict) -> UnitFloat:
        """
        Copy the ``UnitFloat`` and all attributes.

        This method is required because otherwise the ``float.__deepcopy__`` is
        used, which attempts to create a new ``UnitFloat`` class using only 2
        arguments i.e. the ``value``.  ``UnitFloat.__new__`` takes exactly 3
        arguments.

        It simply creates a new ``UnitFloat`` and sets all of its attributes to
        deepcopies of the current attributes (where possible), along with
        updating the ``memo``.

        Parameters
        ----------
        memo : dict
            Dictionary of already deep-copied data.

        Returns
        -------
        UnitFloat
            Deep-copy of `self`.
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

        return self.__repr__()


class UnitNDArray(np.ndarray):
    """
    Subclasses ``ndarray`` so that it contains a ``unit`` attribute.

    ``unit`` attribute is returned when ``__repr__`` or ``__str__`` are called.

    Parameters
    ----------
    shape : tuple of ints
        Shape of created ``array``.
    unit : Unit, str
        A ``Unit`` or a `str` representing the unit.
    dtype : data-type, optional
        Any object that can be interpreted as a NumPy data type.
    buffer : object exposing NumPy buffer interface, optional
        Used to fill the ``array`` with data.
    offset : int, optional
        Offset of ``array`` data in ``buffer``.
    strides : tuple of ints, optional
        Strides of data in memory.
    order : str, optional
        Either ``C`` for row-major or ``F`` for column-major. Default is ``C``.
    """

    def __new__(cls, shape, unit, dtype=float, buffer=None, offset=0,
                strides=None, order=None):
        obj = super().__new__(cls, shape, dtype, buffer, offset, strides, order)
        obj.unit = unit
        return obj

    def __array_finalize__(self, obj):
        self.unit = getattr(obj, 'unit', None)

    @property
    def unit(self) -> None:
        """
        Get or set the ``unit``.

        Either a ``str`` or a ``Unit`` can be passed to the setter.

        Returns
        -------
        Unit
            The ``Unit`` equivalent to the passed ``unit`` parameter.
        """

        return self._unit

    @unit.setter
    def unit(self, value: float):

        if not (isinstance(value, str) or value is None):
            raise TypeError('unit must be a string')
        self._unit = Unit(value)

    def __repr__(self):
        try:
            return super().__repr__() + ' ' + self.unit
        except TypeError:
            return super().__repr__()

    def __str__(self):
        try:
            return super().__str__() + ' ' + self.unit
        except TypeError:
            return super().__str__()


def unit_array(obj, unit: Union[Unit, str], dtype=None) -> UnitNDArray:
    """
    Create a ``UnitNDArray`` from an ``array`` or any nested sequence.

    This mimics the manner in which NumPy creates arrays (although is in Python
    not C), except several arguments are excluded.

    Also, unlike ``np.array(None)``, passing ``obj=None`` to ``unit_array``
    results in `None` being returned. This allows classes to have properties
    with units which can be either have a ``value`` or be undefined.

    Parameters
    ----------
    obj : None or array_like
        An object derived from ``collections.Sequence``. If `None`, then `None`
        is returned.
    unit : Unit, str
        A ``Unit`` or a `str` representing the unit.
    dtype : data-type, optional
        Any object that can be interpreted as a NumPy data type.

    Returns
    -------
    ``UnitArray``
        A ``UnitArray`` object satisfying the specified requirements.
    """

    if obj is None:
        return None

    if not isinstance(unit, str):
        raise TypeError(f'unit must be a string, but was {unit}')

    # Significantly faster to create np.array and view it than to loop
    if not isinstance(obj, np.ndarray):
        obj = np.array(obj, dtype=dtype)

    unit_arr = obj.view(UnitNDArray)
    unit_arr.unit = unit
    return unit_arr


# Update the module scope to include the SYSTEM and units keys
globals().update(SYSTEM)
globals().update(create_units(CODATA_VERSION)[0])
