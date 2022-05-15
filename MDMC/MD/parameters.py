"""A module for the Parameter and Parameters classes

Parameter defines the name and value of each force field parameter, and whether
it is fixed, has constraints or is tied.

Parameters inherits from lists and implements a number of methods for filterting
a sequence of Parameter objects.
"""

import ast
from collections.abc import Iterable
from itertools import chain
import operator
import warnings
import weakref

from MDMC.common.decorators import repr_decorator, unit_decorator, \
    unit_decorator_getter


@repr_decorator('name', 'value', 'unit', 'fixed', 'constraints',
                'interactions_name', 'functions_name', 'tied')
class Parameter:

    """
    A force field parameter which can be fixed or constrained within limits

    The value of a parameter cannot be set if ``fixed==True``.

    Parameters
    ----------
    value : float
        The value of the parameter.
    name :  str
        The name of the parameter.
    fixed : bool
        Whether or not the value can be changed.
    constraints : tuple
        The closed range of the ``Parameter.value``, (lower, upper).
        ``constraints`` must have the same units as ``value``.
    **settings
        ``unit`` (`str`)
            The unit. If this is not provided then the unit will be taken from
            the object passed as ``value``.
    """

    def __init__(self, value, name, fixed=False, constraints=None, **settings):

        self.name = name
        try:
            self.unit = settings['unit'] if 'unit' in settings else value.unit
        except AttributeError:
            self.unit = None
        self.constraints = constraints
        self.value = value
        self.fixed = fixed
        self.interactions_name = None
        self.functions_name = None
        self._interactions = []
        self._tie = None
        self._tie_parameter = None

    @property
    def value(self):
        """
        Get or set the value of the ``Parameter``

        The value will not be changed if it is ``fixed`` or ``tied``, or if it
        is set outside the bounds of ``constraints``

        Returns
        -------
        float
            The value of the ``Parameter``, including if the ``Parameter`` is
            ``tied``

        Warns
        --------
        warnings.warn
            If the ``Parameter`` is ``fixed``.
        warnings.warn
            If the ``Parameter`` is ``tied``.
        """

        if self.tied:
            return self.tie
        return self._value

    @value.setter
    @unit_decorator(unit=None)
    def value(self, value):

        if hasattr(self, 'fixed') and self.fixed:
            warnings.warn("Unable to change fixed parameter")
        elif self.tied:
            warnings.warn("Unable to change tied parameter")
        else:
            if self.constraints is not None:
                self.validate_value(value, self.constraints)
            self._value = value

    @property
    @unit_decorator_getter(unit=None)
    def constraints(self):
        """
        Get or set the constraint of the ``Parameter``

        Returns
        -------
        tuple
            The closed range of the ``Parameter.value``

        Raises
        ------
        ValueError
            If the constraint tuple is not ``(lower, upper)``.
        """

        return self._constraints

    @constraints.setter
    def constraints(self, constraints):

        # Checks if constraints are a 2 element tuple of floats, that the
        # zeroeth element is less than or equal to the first, and that
        # self.value is within them, if it exists
        if constraints is not None:
            if constraints[0] > constraints[1]:
                raise ValueError("Constaints must be (lower, upper)")
            if hasattr(self, 'value'):
                self.validate_value(self.value, constraints)
        self._constraints = constraints

    @property
    def interactions(self):
        """
        Get or append to the parent ``Interaction`` objects for this
        ``Parameter``

        Returns
        -------
        list
            All parent ``Interaction`` objects

        Raises
        ------
        ValueError
            If an added interaction name is not consistent with existing
            interaction names
        ValueError
            If an added ``Interaction`` has a function name not consistent with
            the function names of an existing ``Interaction``
        """

        return [interaction() for interaction in self._interactions]

    @interactions.setter
    def interactions(self, interaction):

        # Test if interaction is of the same type as any interactions already
        # stored
        if self.interactions_name:
            if interaction.name != self.interactions_name:
                raise ValueError('Added interaction name is not consistent with'
                                 ' existing interaction names')
            if interaction.function_name != self.functions_name:
                raise ValueError('Added function name is not consistent with'
                                 ' existing function names')
        else:
            self.interactions_name = interaction.name
            self.functions_name = interaction.function_name

        self._interactions.append(weakref.ref(interaction))

    @property
    def tie(self):
        """
        Get the ``value`` of a the ``Parameter`` that this ``Parameter`` is tied
        to

        Returns
        -------
        float
            The ``value`` of the ``tied`` ``Parameter``
        """
        # pylint: disable=eval-used
        # eval use is generally bad
        # but the safe alternative (ast.literal_eval) creates malformed nodes

        if self._tie is None:
            return None
        return eval(compile(self._tie, '', 'eval'))

    @property
    def tied(self):
        """
        Get whether this ``Parameter`` is tied

        Returns
        -------
        bool
            `True` if this ``Parameter`` is tied to another ``Parameter``, else
            `False`
        """

        return bool(hasattr(self, 'tie') and self.tie is not None)

    def set_tie(self, parameter, expr):
        """
        This ``ties`` the ``Parameter.value`` to the ``value`` of another
        ``Parameter``

        Parameters
        ---------
        parameter : Parameter
            The ``Parameter`` to tie to
        expr : str
            A mathematical expression

        Examples
        --------
        To set the ``Parameter.value`` to ``p1.value * 2``::

        >>> Parameter.set_tie(p1, "* 2")
        """

        self._tie_parameter = weakref.ref(parameter)
        self._tie = ast.parse(
            'self._tie_parameter().value' + expr, mode='eval')

    def __str__(self):

        condition = ('Fixed ' if self.fixed else 'Tied ' if self.tied else
                     'Constrained ' if self.constraints is not None else '')
        function = self.functions_name + ' ' if self.functions_name else ''
        return '{0}{_value} {1}{name}'.format(condition, function,
                                              **self.__dict__)

    def __getitem__(self, key):

        return self.__getattribute__(key)

    def __setitem__(self, key, value):

        self.__setattr__(key, value)

    @staticmethod
    def validate_value(value, constraints):
        """
        Validates the ``Parameter.value`` by testing if it is within the
        ``constraints``

        Parameters
        ----------
        values : float
            The value of the ``Parameter``

        Raises
        ------
        ValueError
            If the ``value`` is not within the ``constraints``
        """

        if value < constraints[0] or value > constraints[1]:
            raise ValueError("Value must be within constraints")

    # comparison operator so parameters are always in the same order on refinement headings
    def __lt__(self, other):
        return self.name < other.name


class Parameters(list):

    """
    A `list-like` object where every element is a ``Parameter``, which contains
    a number of helper methods for filtering
    """

    def __getitem__(self, key):

        item = super().__getitem__(key)
        if isinstance(key, slice):
            return self.__class__(item)
        return item

    def filter(self, predicate):
        """
        Filters using a predicate

        Parameters
        ----------
        predicate : function
            A function that returns a `bool` which takes a ``Parameter`` as an
            argument.

        Returns
        -------
        Parameters
            The ``Parameter`` objects which meet the condition of the predicate
        """

        return Parameters(filter(predicate, self))

    def filter_name(self, name):
        """
        Filters by ``name``

        Parameters
        ----------
        name : str
            The ``name`` of the ``Parameter`` objects to return.

        Returns
        -------
        Parameters
            The ``Parameter`` objects with ``name``
        """

        return Parameters(filter(lambda p: p.name == name, self))

    def filter_value(self, comparison, value):
        """
        Filters by ``value``

        Parameters
        ----------
        comparison : str
            A `str` representing a comparison operator, ``'>'``, ``'<'``,
            ``'>='``, ``'<='``, ``'=='``, ``'!='``.
        value : float
            A `float` with which ``Parameter`` values are compared, using the
            ``comparison`` operator.

        Returns
        -------
        Parameters
            The ``Parameter`` objects which return a `True` when their values
            are compared with ``value`` using the ``comparison`` operator
        """

        ops = {'>': operator.gt,
               '<': operator.lt,
               '>=': operator.ge,
               '<=': operator.le,
               '==': operator.eq,
               '!=': operator.ne}

        return Parameters(filter(lambda p: ops[comparison](p.value, value), self))

    def filter_interaction(self, interaction_name):
        """
        Filters based on the name of the ``Interaction`` of each ``Parameter``

        Parameters
        ----------
        interaction_name : str
            The name of the ``Interaction`` of ``Parameter`` objects to return,
            for example ``'Bond'``.

        Returns
        -------
        Parameters
            The ``Parameter`` objects which have an ``Interaction`` with the
            specified ``interaction_name``
        """

        return Parameters(filter(lambda p: p.interactions_name == interaction_name,
                                 self))

    def filter_function(self, function_name):
        """
        Filters based on the name of the ``InteractionFunction`` of each
        ``Parameter``

        Parameters
        ----------
        function_name : str
            The name of the ``InteractionFunction`` of ``Parameter`` objects to
            return, for example ``'LennardJones'`` or ``'HarmonicPotential'``.

        Returns
        -------
        Parameters
            The ``Parameter`` objects which have a ``function`` with the
            specified ``function_name``
        """

        return Parameters(filter(lambda p: p.functions_name == function_name, self))

    def filter_atom_attribute(self, attribute, value):
        """
        Filters based on the attribute of ``Atom`` objects which have each
        ``Parameter`` applied to them


        Parameters
        ----------
        attribute : str
            An attribute of an ``Atom``. Attributes to match to must be either
            `float` or str.
        value : str, float
            The value of the ``Atom`` ``attribute``.

        Returns
        -------
        Parameters
            The ``Parameter`` objects which are applied to an ``Atom`` object
            which has the specified ``value`` of the specified ``attribute``
        """

        def flatten(iterable):
            for element in iterable:
                if isinstance(element, Iterable):
                    yield from flatten(element)
                else:
                    yield element

        return Parameters(filter(lambda p:
                                 value in [getattr(atom, attribute)
                                           for int in p.interactions
                                           for atom
                                           in flatten(int.atoms)],
                                 self))

    def filter_structure(self, structure_name):
        """
        Filters based on the name of the ``Structure`` to which each
        ``Parameter`` applies

        Parameters
        ----------
        structure_name : str
            The name of a ``Structure``.

        Returns
        -------
        Parameters
            The ``Parameter`` objects which are applied to a ``Structure``
            which has the specified ``zstructure_name``
        """

        def check_structure_name(parameter):
            """
            Checks the name of all structures

            Returns
            -------
            list
                A `list` of `str` with the names of ``Structure`` objects
            """

            # Recursively add structure.name to structure_names set until the
            # structure is the top level structure
            structure_names = set()

            def add_name(structure):
                structure_names.add(structure.name)
                if structure.top_level_structure == structure:
                    return
                add_name(structure.parent)

            for inter in parameter.interactions:
                for atom in chain.from_iterable(inter.atoms):
                    add_name(atom)
            return structure_name in structure_names

        return Parameters(filter(check_structure_name, self))
