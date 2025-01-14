"""A module for the Parameter and Parameters classes

Parameter defines the name and value of each force field parameter, and whether
it is fixed, has constraints or is tied.

Parameters inherits from lists and implements a number of methods for filterting
a sequence of Parameter objects.
"""

from __future__ import annotations

import ast
import logging
import operator
import re
import warnings
import weakref
from collections.abc import Callable, Iterable
from itertools import chain, count
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np

from MDMC.common.decorators import repr_decorator, unit_decorator, unit_decorator_getter

if TYPE_CHECKING:
    from MDMC.MD.interactions import Interaction


@repr_decorator('ID', 'type', 'value', 'unit', 'fixed', 'constraints',
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

    # each Parameter has a unique ID, so they can be distinguished
    _ID_generator = count(start=1, step=1)

    def __init__(self, value, name, fixed=False, constraints=None, **settings):

        self.ID = self._generate_ID()
        self.name = name + f" (#{self.ID})"
        self.type = name
        self.unit = settings.get('unit', getattr(value, 'unit', None))
        self.constraints = constraints
        self.value = value
        self.fixed = fixed
        self.interactions_name = None
        self.functions_name = None
        self._interactions = []
        self._tie = None
        self._tie_parameter = None

    @property
    def value(self) -> float:
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
    def value(self, value: float) -> None:

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
    def constraints(self) -> tuple:
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
    def constraints(self, constraints: tuple) -> None:

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
    def interactions(self) -> list:
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
    def interactions(self, interaction: Interaction) -> None:

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
    def tie(self) -> float | None:
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
    def tied(self) -> bool:
        """
        Get whether this ``Parameter`` is tied

        Returns
        -------
        bool
            `True` if this ``Parameter`` is tied to another ``Parameter``, else
            `False`
        """

        return bool(hasattr(self, 'tie') and self.tie is not None)

    def set_tie(self, parameter: Parameter, expr: str) -> None:
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

    @classmethod
    def _generate_ID(cls) -> int:
        """Generates a unique ID for the Parameter that has just been created."""
        return next(cls._ID_generator)

    def __str__(self) -> str:

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
    def validate_value(value: float, constraints: tuple) -> None:
        """
        Validates the ``Parameter.value`` by testing if it is within the
        ``constraints``

        Parameters
        ----------
        values : float
            The value of the ``Parameter``
        constraints: tuple
            A 2-tuple of the lower and upper constraints respectively.

        Raises
        ------
        ValueError
            If the ``value`` is not within the ``constraints``
        """

        if value < constraints[0] or value > constraints[1]:
            raise ValueError(f"Value must be within constraints, \
                            value is: {value}, constraints are: {constraints}")

    # comparison operator so parameters are always in the same order on refinement headings
    def __lt__(self, other):
        return self.name < other.name


class Parameters(dict):
    """
    A `dict-like` object where every element is a ``Parameter`` indexed by name,
    which contains a number of helper methods for filtering.

    Although ``Parameters`` is a `dict`, it should be treated like a `list` when writing to it;
    i.e. initialise it using a `list` and use `append` to add to it. These parameters can
    then be accessed by their name as a key.

    In short; Parameters writes like a list and reads like a dict.

    Parameters
    ----------
    init_parameters: ``Parameter`` or `list` of ``Parameter``s, optional, default None
        The initial ``Parameter`` objects that the ``Parameters`` object contains.

    Attributes
    ----------
    array: np.ndarray
        An alphabetically-sorted numpy array of the ``Parameter``s stored in this object.
    """

    def __init__(self, init_parameters: list[Parameter] | Parameter | None = None):
        super().__init__()
        if init_parameters is not None:
            init_parameters = self._check_input(init_parameters)
            self.append(init_parameters)

    def __setitem__(self, key: str, value: Parameter) -> NoReturn:
        # disable this method to ensure parameter keys are always the parameter name
        raise TypeError("Parameters should be added to using Parameters.append(parameter), "
                        "with a parameter or list of parameters as your argument.")

    def __getitem__(self, key: str) -> list[Parameter] | Parameter:
        try:
            return super().__getitem__(key)
        except KeyError as error:
            # see if the key passed was a parameter name with no ID, and catch the error
            # by getting the first parameter with that name
            r = re.compile(rf"{key} \(#[0-9]+\)")
            matching_parameters = list(filter(r.match, list(self.keys())))
            if matching_parameters:
                if len(matching_parameters) > 1:
                    warnings.warn("Calling a parameter name with no ID returns a "
                                  "list of all parameters with that name; "
                                  "this may create inconsistent behaviour!")
                    #pylint: disable=super-with-arguments
                    # for some reason when we run it without arguments,
                    # it complains in jupyter notebooks
                    # see http://thomas-cokelaer.info/blog/2011/09/382/
                    return sorted([super(Parameters, self).__getitem__(p)
                                   for p in matching_parameters],
                                   key=lambda p: p.ID)
                return super().__getitem__(matching_parameters[0])
            raise KeyError from error

    def append(self, parameters: list[Parameter] | Parameter) -> None:
        """
        Appends a ``Parameter`` or list of ``Parameter``s to the dict,
        with the parameter name as its key.

        Parameters
        ----------
        parameters: ``Parameter`` or `list` of ``Parameter``s
            The parameter(s) to be added to the dict.
        """
        parameters = self._check_input(parameters)

        for parameter in parameters:
            super().__setitem__(parameter.name, parameter)


    @property
    def as_array(self) -> np.ndarray:
        """
        The parameters in the object as a sorted numpy array.

        Returns
        -------
        np.ndarray
            An alphabetically-sorted array of parameter values in the object.
        """

        return np.array(sorted(list(self.values()), key=lambda p: p.name))

    def filter(self, predicate: Callable[[Parameter], bool]) -> Parameters:
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

        return Parameters(list(filter(predicate, list(self.values()))))

    def filter_name(self, name: str) -> Parameters:
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

        return self.filter(lambda p: name in p.name)

    def filter_value(self, comparison: str, value: float) -> Parameters:
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

        return self.filter(lambda p: ops[comparison](p.value, value))

    def filter_interaction(self, interaction_name: str) -> Parameters:
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

        return self.filter(lambda p: p.interactions_name == interaction_name)

    def filter_function(self, function_name: str) -> Parameters:
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

        return self.filter(lambda p: p.functions_name == function_name)

    def filter_atom_attribute(self, attribute: str, value: str | float) -> Parameters:
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

        return self.filter(lambda p:
                           value in [getattr(atom, attribute)
                                     for interaction in p.interactions
                                     for atom in flatten(interaction.atoms)])

    def filter_structure(self, structure_name: str) -> Parameters:
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
            which has the specified ``structure_name``
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

        return self.filter(check_structure_name)

    def log_parameters(self) -> None:
        """Logs all Parameters by ID"""

        LOGGER = logging.getLogger(__name__)
        msg = "List of all parameters with ID: \n"
        for parameter in self.values():
            msg += f"{parameter.repr()}"

        LOGGER.info(msg)
        print("Details on which Parameter corresponds to each ID have been written to the log.")

    @staticmethod
    def _check_input(x: Any) -> list[Parameter]:
        """
        Ensures that input to a Parameters object is in the correct form.

        Raises an error if the input is not either a Parameter
        (in which case it is turned into a list, so it can be fed into an iteration loop)
        or a list of Parameters.

        Parameters
        ----------
        x: Any
            The object to be sanitised.

        Returns
        -------
        list[Parameter]
            Returns x if x is a list of Parameters, or [x] if x is a Parameter
            (so it can be iterated over)

        Raises
        ------
        TypeError
            If the object is not a Parameter or list of Parameters (including
            if it is a list that contains a non-Parameter object)
        """
        if isinstance(x, Parameter):
            return [x]
        if isinstance(x, list) and all(isinstance(i, Parameter) for i in x):
            return x

        raise TypeError("Input into a Parameters object must be either a Parameter "
                        "or a list of Parameters.")
