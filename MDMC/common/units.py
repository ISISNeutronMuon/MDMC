"""Module for all unit definitions and operations

This includes defining units used in MDMC, converting units, and subclassing
data strucures (float, NumPy array) so that they have a unit attribute

AUTHOR :    Thomas Farmer        START DATE :    12/12/2018, 11:06:51"""

import numpy as np


class UnitFloat(float):

    """
    Subclasses float so that it contains a unit attribute which is returned when
    __repr__ or __str__ are called
    """

    def __new__(cls, value, unit):

        return float.__new__(cls, value)

    def __init__(self, value, unit):

        float.__init__(value)
        self.unit = unit

    def __repr__(self):

        return repr(self.real) + ' ' + self.unit


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


def unit_array(obj, unit, dtype=None):

    """
    Helper function for creating a UnitNDArray from an array or any nested
    sequence

    This mimics the manner in which numpy creates arrays (although is in python
    not C), except several arguments are excluded

    Arguments:
    object - an array or array-like object (e.g. any object derived from
    collections.Sequence)
    unit - a string specifying the unit of the array
    dtype - the desired data-type for the array
    """

    # Significantly faster to create np.array and view it than to loop
    if not isinstance(obj, np.ndarray):
        obj = np.array(obj, dtype=dtype)

    unit_arr = obj.view(UnitNDArray)
    unit_arr.unit = unit
    return unit_arr
