"""Module which defines decorators

AUTHOR :    Thomas Farmer        START DATE :    12/12/2018, 10:46:48"""

from MDMC.common.units import UnitFloat, unit_array


def unit_decorator(unit):

    """
    Decorates property.setter methods to add units to the values which are
    passed to them. These units are displayed when either repr or str is called
    for the corresponding property.getter method.

    Suitable for use with setter methods that either take floats (or objects
    that can be cast to floats), or NumPy arrays (or objects that can be cast
    to NumPy arrays)

    Arguments:
    unit - string specifying the unit
    """

    def decorator(func):
        def wrapper(self, value):
            try:
                return func(self, UnitFloat(value, unit))
            except TypeError:
                return func(self, unit_array(value, unit))
        return wrapper
    return decorator
