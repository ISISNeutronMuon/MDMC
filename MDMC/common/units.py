"""Module for all unit definitions and operations

This includes defining units used in MDMC, converting units, and subclassing
data strucures (float, NumPy array) so that they have a unit attribute.  This
style follows that of the Atomic Simulation Environment.

AUTHOR :    Thomas Farmer        START DATE :    12/12/2018, 11:06:51"""

from copy import deepcopy

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
    """

    def __mul__(self, other):

        """
        Appends a single space and other to the unit string
        """

        return self.__class__(self + ' ' + other)

    def __div__(self, other):

        """
        Appends ' / ' and other to the unit string
        """

        return self.__class__(self + ' / ' + other)

    def __pow__(self, other):

        """
        Appends ' ^ ' and other to the unit string
        """

        try:
            return self.__class__(self + ' ^ ' + other)
        except TypeError:
            return self.__class__(self + ' ^ ' + str(other))


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

    # Energy

    return units

class UnitFloat(float):

    """
    Subclasses float so that it contains a unit attribute which is returned when
    __repr__ or __str__ are called

    Attributes:
    unit - a string specifying the unit

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

    def __deepcopy__(self, memo):

        """
        Copies the UnitFloat and all attributes

        This method is required because otherwise the float.__deepcopy__ is
        used, which attempts to create a new UnitFloat class using only 2
        argument i.e. the value.  UnitFloat.__new__ takes exactly 3 arguments.

        It simply creates a new UnitFloat and sets all of its attributes to
        deepcopies of the current attributes (along with updating the memo).
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
    unit - a string specifying the unit of the array
    dtype - the desired data-type for the array
    """

    if obj is None:
        return None

    # Significantly faster to create np.array and view it than to loop
    if not isinstance(obj, np.ndarray):
        obj = np.array(obj, dtype=dtype)

    unit_arr = obj.view(UnitNDArray)
    unit_arr.unit = unit
    return unit_arr


# Update the module scope to include the SYSTEM and units keys
globals().update(SYSTEM)
globals().update(create_units(CODATA_VERSION))
