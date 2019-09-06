"""Module which defines decorators"""

from functools import wraps
import textwrap

from MDMC.common.units import UnitFloat, unit_array


def unit_decorator(unit):

    """
    Decorates property.setter methods to add units

    Adds units to the values passed to property.setter methods. These units are
    displayed when either repr or str is called for the corresponding
    property.getter method.

    Suitable for use with setter methods that either take floats (or objects
    that can be cast to floats), or NumPy arrays (or objects that can be cast
    to NumPy arrays)

    Parameters
    ----------
    unit : string or None
        The unit applied to the property. If None then self.unit is used, which
        enables classes to have properties with units defined at runtime.

    Returns
    -------
    function
        A property.setter function with a value parameter which has a unit.

    Example
    -------
    Add a unit_decorator to the position property::

        >>> Class Atom(StructuralUnit):
        ...
        ...     @property
        ...     def position(self):
        ...         return self._position
        ...
        ...     @position.setter
        ...     @unit_decorator(unit=Unit('Ang'))
        ...     def position(self, value):
        ...         self._position = value
    """

    # Ignore pylint warning for decorator inner function docstrings
    #pylint: disable=missing-docstring
    def decorator(func):
        def unit_creator(self, value, unit):
            try:
                return func(self, UnitFloat(value, unit))
            except TypeError:
                return func(self, unit_array(value, unit))

        def wrapper(self, value):
            if unit is None:
                return unit_creator(self, value, self.unit)
            return unit_creator(self, value, unit)
        return wrapper
    return decorator


def unit_decorator_getter(unit):

    """
    Decorates property.getter methods to add units

    Adds units to the return values of property.getter methods. These units are
    displayed when either repr or str is called.

    Suitable for use with setter methods that either take floats (or objects
    that can be cast to floats), or NumPy arrays (or objects that can be cast
    to NumPy arrays). This method exists for properties which have no setter
    method.

    Parameters
    ----------
    unit : string or None
        The unit applied to the property. If None then self.unit is used, which
    enables classes to have properties with units defined at runtime.

    Returns
    -------
    function
        A property.getter function with a return type which has a unit (e.g.
        UnitFloat or UnitArray)

    Example
    -------
    Add a unit_decorator_getter to the volume property::

        >>> Class Universe(object):
        ...
        ...     @property
        ...     @unit_decorator_getter(unit=Unit('Ang') ^ 3)
        ...     def volume(self):
        ...         return self.dims ** 3
    """

    # Ignore pylint warning for decorator inner function docstrings
    #pylint: disable=missing-docstring
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


def set_func_docstring(docstring):

    """
    Decorator for setting the docstring of a function or method.

    The new docstring is text wrapped to ensure that the line length is valid.
    It is assumed that the specified docstring has the correct indentations.

    Parameters
    ----------
    docstring : str
        The new docstring for the function or method

    Returns
    -------
    A decorator which sets the docstring of a function or method
    """

    # Ignore pylint warning for decorator inner function docstrings
    #pylint: disable=missing-docstring
    def decorator(func):
        # docstring must be set outside of wrapper. This means that
        # functools.wraps can be used to preserve the docstring after the
        # function has been wrapped.
        func.__doc__ = _wrap_docstring(docstring, 80)
        @wraps(func)
        def wrapper(*args, **settings):
            func(*args, **settings)
        return wrapper
    return decorator


def mod_func_docstring(replacements):

    """
    Decorator for modifying the docstring of a function or method.

    This is done by replacing specified substrings. After replacement the
    docstring is text wrapped to ensure that line length and indentations are
    preserved.

    Parameters
    ----------
    replacements : dict
        {old:new} pairs where old is a str in the docstring which will be
        replaced, and new is the str it should be replaced with.

    Returns
    -------
    A decorator which modifies the docstring of a function or method
    """

    # Ignore pylint warning for decorator inner function docstrings
    #pylint: disable=missing-docstring
    def decorator(func):
        # docstring must be modified outside of wrapper. This means that
        # functools.wraps can be used to preserve the docstring after the
        # function has been wrapped.
        for old, new in replacements.items():
            func.__doc__ = func.__doc__.replace(old, new)
        func.__doc__ = _wrap_docstring(func.__doc__, 80)
        @wraps(func)
        def wrapper(*args, **settings):
            func(*args, **settings)
        return wrapper
    return decorator


def _wrap_docstring(docstring, line_length):

    """
    Wraps a docstring to a specific line length.

    This maintains any indentation which exists at the start of a line

    Parameters
    ----------
    docstring : str
        The docstring to be wrapped
    line_length : int
        The maximum line length of the docstring before it is wrapped

    Returns
    -------
    str
        The wrapped docstring
    """

    wrapped = []
    for line in docstring.split('\n'):
        if len(line) > line_length:
            indent = (len(line) - len(line.strip())) * ' '
            wrap = textwrap.wrap(line, line_length, subsequent_indent=indent)
            wrap = ['\n' + element for element in wrap]
            wrapped += wrap
        else:
            wrapped.append('\n' + line)
    return ''.join(wrapped)
