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

    Example:

    Class Atom(StructuralUnit):

        @property
        def position(self):
            return self._position

        @position.setter
        @unit_decorator(unit='Ang')
        def position(self, value):
            self._position = value
    """

    def decorator(func):
        def wrapper(self, value):
            try:
                return func(self, UnitFloat(value, unit))
            except TypeError:
                return func(self, unit_array(value, unit))
        return wrapper
    return decorator


def unit_decorator_getter(unit):

    """
    Decorates property.getter methods to add units to the return value. These
    units are displayed when either repr or str is called.

    Suitable for use with setter methods that either take floats (or objects
    that can be cast to floats), or NumPy arrays (or objects that can be cast
    to NumPy arrays). This method exists for properties which have no setter
    method.

    Arguments:
    unit - string specifying the unit or None. If None then self.unit is used,
    which enables classes to have properties with units defined at runtime.

    Example:

    Class Universe(object):

        @property
        @unit_decorator_getter(unit='Ang ^ 3')
        def volume(self):
            return self.dims ** 3
    """

    def decorator(func):
        def unit_creator(self, unit):
            try:
                return UnitFloat(func(self), unit)
            except TypeError:
                return unit_array(func(self), unit)

        def wrapper(self):
            if unit is None:
                return unit_creator(self, self.unit)
            return unit_creator(self, unit)
        return wrapper
    return decorator
