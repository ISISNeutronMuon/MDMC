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

        >>> Class Universe:
        ...
        ...     @property
        ...     @unit_decorator_getter(unit=Unit('Ang') ^ 3)
        ...     def volume(self):
        ...         return self.dimensions ** 3
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
    function
        A decorator which sets the docstring of a function or method

    Example
    -------
    To dynamically set the docstring of a function:

        .. highlight:: python
        .. code-block:: python

            @set_func_docstring("This is the new docstring")
            def function():
                \"\"\"
                This docstring will be replaced
                \"\"\"
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

    While this can be used for replacements in equations, care must be taken
    to ensure that wrapping does not cause line breaks in invalid places in the
    Latex.

    Parameters
    ----------
    replacements : dict
        {old:new} pairs where old is a str in the docstring which will be
        replaced, and new is the str it should be replaced with.

    Returns
    -------
    function
        A decorator which modifies the docstring of a function or method

    Example
    -------
    To dynamically modify the docstring of a function so 'this' is replaced
    with 'that':

        .. highlight:: python
        .. code-block:: python

            @set_func_docstring({'this':'that'})
            def function():
                \"\"\"
                The word this will be replaced
                \"\"\"
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

    This maintains any indentation which exists at the start of a line. While
    equations should not be affected by this wrapping, it is recommended that
    docstrings with .. math:: are visually checked after wrapping.

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

    Raises
    ------
    ValueError
        If any indent has more characters than the line_length, as the wrapping
        cannot then preserve the correct indent
    """

    wrapped = []
    prev_line = None
    prev_indent = None

    for line in docstring.split('\n'):
        # Get indent of right length for line
        indent = (len(line) - len(line.strip())) * ' '
        if len(indent) >= line_length:
            raise ValueError('The line length is shorter than one or more'
                             ' indents')
        # If previous line was wrapped and has same length of indent, then
        # prepend it to this line
        if prev_line is not None:
            if prev_indent == indent and not '.. math::' in line:
                line = prev_line + ' ' + textwrap.dedent(line)
            else:
                wrapped.append('\n' + prev_line)
        # Wrap line if the length is greater than the line length
        if len(line) > line_length:
            wrap = textwrap.wrap(line, line_length, subsequent_indent=indent)
            prev_line = wrap[-1]
            wrap = ['\n' + element for element in wrap[:-1]]
            wrapped += wrap
            prev_indent = indent
        else:
            prev_line = None
            wrapped.append('\n' + line)
    # If last line in docstring was wrapped, append this to the array
    if prev_line is not None:
        wrapped.append('\n' + prev_line)
    # Accounting for case of docstring starting on line after """
    if wrapped[0] == '\n':
        del wrapped[0]
    # Accounting for case of docstring starting on same line as """
    elif wrapped[0][0] == '\n':
        wrapped[0] = wrapped[0][1:]

    return ''.join(wrapped)
