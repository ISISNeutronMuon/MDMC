"""A module for storing atomic interaction functions

Contains class InteractionFunction from which all interaction function classes
must derive.  All functions describing atomic interactions must be added to this
module in order to be called by a universe.  If needed, the interaction function
classes can be extended to contain actual function definitions.

Contains class Parameter, which defines the name and value of each
parameter which belongs to an InteractionFunction, and whether the parameter is
fixed, has constraints or is tied.

Contains filters for filtering list of parameters based on a predicate."""

import ast
import functools
from inspect import getargspec, getmembers
from itertools import chain, zip_longest
import operator
import warnings
import weakref

import numpy as np

from MDMC.common.decorators import unit_decorator, unit_decorator_getter
from MDMC.common import units


class Parameter:

    """
    A force field parameter which can be fixed or constrained within limits

    The value of a parameter cannot be set if fixed=True.  The constraints
    specif

    As __repr__ returns a string of a dictionary of the public attributes (so
    that these are easy to view), for consistency this class implements some
    dictionary methods.

    Parameters
    ----------
    value : float
        The value of the parameter.
    name :  str
        The name of the parameter.
    fixed : bool
        Whether or not the value can be changed.
    constraints : tuple
        The closed range of the Parameter value, (lower, upper). Constraints
        must have the same units as value.
    **settings
        unit : str
            The unit. If this is not provided then the unit will be taken from
            the object passed as value.
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

    @property
    def value(self):

        """
        Get or set the value of the Parameter

        The value will not be changed if it is fixed or tied, or if it is set
        outside the bounds of a constraint

        Returns
        -------
        float
            The value of the Parameter, including if the parameter is tied

        Warns
        --------
        warnings.warn
            If the Parameter is fixed.
        warnings.warn
            If the Parameter is tied.
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
        Get or set the constraint of the Parameter

        Returns
        -------
        tuple
            The closed range of the Parameter value

        Raises
        ------
        ValueError
            If the constraint tuple is not (lower, upper).
        """

        return self._constraints

    @constraints.setter
    def constraints(self, constraints):

        ### Checks if constraints are a 2 element tuple of floats, that the
        ### zeroeth element is less than or equal to the first, and that
        ### self.value is within them, if it exists
        if constraints is not None:
            if constraints[0] > constraints[1]:
                raise ValueError("Constaints must be (lower, upper)")
            if hasattr(self, 'value'):
                self.validate_value(self.value, constraints)
        self._constraints = constraints

    @property
    def interactions(self):

        """
        Get or append to the parent Interaction objects for this Parameter

        Returns
        -------
        list
            All parent Interaction objects

        Raises
        ------
        ValueError
            If an added interaction name is not consistent with existing
            interaction names
        ValueError
            If an added interaction has a function name not consistent with
            the function names of existing interactions
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
        Get the value of a the Parameter object that this Parameter is tied to

        Returns
        -------
        float
            The value of the tied Parameter
        """

        if self._tie is None:
            return None
        else:
            return eval(compile(self._tie, '', 'eval'))

    @property
    def tied(self):

        """
        Get whether this Parameter is tied

        Returns
        -------
        bool
            True if this Parameter is tied to another Parameter, else False
        """

        if hasattr(self, 'tie') and self.tie is not None:
            return True
        else:
            return False

    def set_tie(self, parameter, expr):

        """
        This ties the parameter's value to the value of another parameter

        Parameters
        ---------
        parameter : Parameter
            The Parameter to tie to
        expr : str
            A mathematical expression

        Examples
        --------
        To set the Parameters return value to p1.value * 2::

        >>> Parameter.set_tie(p1, "* 2")
        """

        self._tie_param = weakref.ref(parameter)
        self._tie = ast.parse('self._tie_param().value' + expr, mode='eval')

    def __repr__(self):

        """
        Returns
        -------
        str
            The Parameter name and a dictionary containing properties and their
            values, except self.tie and self.interactions
        """

        return self._get_attr_strings(['tie', 'interactions'])

    def __getitem__(self, key):

        return self.__getattribute__(key)

    def __setitem__(self, key, value):

        self.__setattr__(key, value)

    def validate_value(self, value, constraints):

        """
        Validates the parameter value by testing if it is within the constraints

        Raises
        ------
        ValueError
            If the value is not within the constraints
        """

        if value < constraints[0] or value > constraints[1]:
            raise ValueError("Value must be within constraints")

    def _get_attr_strings(self, excluded=[]):

        """
        Returns
        -------
        str
            The Parameter name and a dictionary containing properties and their
            values
        """

        # Determine which attributes are in the form of properties
        properties = getmembers(self.__class__,
                                lambda o: isinstance(o, property))
        rpr = {p[0]:getattr(self, p[0]) for p in properties
               if p[0] not in excluded}

        return '{name} = {rpr}'.format(name=self.name.replace('_', ' '),
                                       rpr=rpr)


class InteractionFunction:

    """
    Base class for interaction functions, which can be user supplied

    Parameters
    ---------
    val_dict : dict
        name:value pairs. Currently this must be ordered alphabetically. Value
        must either be a object with a value and a unit (e.g. a UnitFloat
        object), or a (float, str) tuple, where float is the value and str is
        the unit.
    """

    def __init__(self, val_dict):

        # locals which are excluded from Parameter creation
        excluded = ['self', 'settings', '__class__']
        params = []
        for name, value in val_dict.items():
            if name not in excluded:
                param = Parameter(value, name)
                params.append(param)
                # Create an attribute with the same name as the Parameter
                setattr(self, param.name, param)
        self.params = params

    @property
    def params(self):

        """
        Get or set the array of Parameters

        On setting the Parameters, they are ordered alphabetically by
        Parameter.name

        Returns
        -------
        np.ndarray
            A NumPy array of Parameters
        """

        return self._params

    @params.setter
    def params(self, value):

        self._params = np.array(sorted(value, key=lambda p: p.name))

    @property
    def params_values(self):

        """
        Get the values for all Parameters

        Returns
        -------
        np.ndarray
            A NumPy array of values for all Parameters
        """

        return np.array([p.value for p in self.params])

    @property
    def name(self):

        """
        Get the name of the class of the InteractionFunction

        Returns
        -------
        str
            The class name
        """

        return self.__class__.__name__

    def set_params_interactions(self, interaction):

        """
        Sets the parent interaction for all Parameters

        Parameters
        ----------
        interaction : Interaction
            An interaction to set as the parent of all the Parameters
        """

        for param in self.params:

            param.interactions = interaction


def inter_func_decorator(*param_units):

    """
    Decorates a method to add units to all non-keyword arguments

    Designed for adding units to parameters of __init__ method for subclasses of
    InteractionFunction.

    Parameters
    ----------
    *param_units
        one or more str or units.Unit, where each str (or Unit) is a unit which
        is applied to the corresponding value passed to the decorated method. If
        one of the values is unitless, pass None at the corresponding index in
        *param_units.

    Examples
    --------
    The following adds units of 'Ang' to parameter alpha, units of 's' to the
    parameter beta, and units of 'atm' to the parameter gamma:

        .. highlight:: python
        .. code-block:: python

            @inter_func_decorator('Ang', 's', 'Pa')
            def __init__(self, alpha, beta, gamma):
                ...

    If one of the parameters is unitless, this can be set with None (in which
    case the returned type will be the same as the original value i.e. a
    UnitFloat or UnitNDArray will not be created). So to set epsilon as
    unitless:

        .. highlight:: python
        .. code-block:: python

            @inter_func_decorator('arb', None, 'deg')
            def __init__(self, delta, epsilon, gamma):
                ...
    """

    # Ignore pylint warning for decorator inner function docstrings
    #pylint: disable=missing-docstring
    def decorator(func):
        def unit_creator(value, unit):
            # If no unit is provided, assume unitless and just return value
            if unit is None:
                return value
            # try/except to determine whether value is float or array
            try:
                return units.UnitFloat(value, unit)
            except TypeError:
                return units.unit_array(value, unit)

        @functools.wraps(func)
        def wrapper(self, *values, **settings):
            # Use zip to associate each value in *values with the corresponding
            # unit in *param_units. unit_creator return a UnitFloat or
            # UnitNDArray with this unit, or returns the original value if the
            # unit is None.
            return func(self, *[unit_creator(value, unit) for value, unit in
                                zip(values, param_units)], **settings)
        return wrapper
    return decorator


class Buckingham(InteractionFunction):

    r"""
    The Buckingham potential (in units of kJ mol^-1) for the interaction of
    2 atoms at distance r (in Ang) has the form:

    .. math::

        {\Phi _{12}(r)=A\exp \left(-Br\right)-{\frac {C}{r^{6}}}}

    Parameters
    ----------
    A : float
        The Buckingham parameter A in units of kJ mol^-1
    B : float
        The Buckingham parameter B in units of Ang^-1
    C : float
        The Buckingham parameter C in units of Ang^6 kJ mol^-1
    """

    @inter_func_decorator(units.ENERGY, units.LENGTH ** -1,
                          units.LENGTH ** 6 * units.ENERGY)
    def __init__(self, A, B, C):

        super().__init__(locals())


class Coulomb(InteractionFunction):

    r"""
    Coulomb interaction for charged particles:

    .. math::

        E = \frac{Cq_{i}q_{j}}{r}

    Parameters
    ----------
    charge : float
        The charge in units of e
    """

    @inter_func_decorator(units.CHARGE)
    def __init__(self, charge):

        super().__init__(locals())


class HarmonicPotential(InteractionFunction):

    r"""
    Harmonic potential for bond stretching, and angular and improper dihedral
    vibration, with the form:

    .. math::

        E = K(r-r_0)^2

    As HarmonicPotential can be used with several different Interaction types,
    the Interaction type must be specified, so that the correct units can be
    assigned to the equilibrium_state and potential_strength parameters.

    Parameters
    ----------
    equilibrium_state : float
        The equilibrium state of the object in either Ang or degrees, depending
        on the interaction_type passed.
    potential_strength : float
        The potential strength in units of kJ mol^-1 Ang^-2 (linear) or
        kJ mol^-1 rad^-2 (angular/improper), depending on the interaction_type
        passed.
    **settings
        interaction_type : str
            A str specifying either 'bond', 'angle' or 'improper'. This assigns
            the correct units to equilibrium_state and potential_strength. This
            keyword must be passed.

    Raises
    ------
    ValueError
        The interaction_type must be 'bond', 'angle', or 'improper'
    TypeError
        An interaction_type of 'bond', 'angle', or 'improper' must be passed

    Example
    -------
    The following result creates a HarmonicPotential for a Bond interaction::

        HarmonicPotential(1.0, 2.0, interaction_type='bond')
    """

    def __new__(cls, equilibrium_state, potential_strength, **settings):

        # interaction_type is a required keyword, but has to be passed through
        # settings so that it can be correctly passed in inter_func_decorator
        try:
            if settings['interaction_type'].lower() == 'bond':
                eq_unit = units.LENGTH
                pot_unit = units.ENERGY / units.LENGTH ** 2
            elif settings['interaction_type'].lower() in ('angle', 'improper'):
                eq_unit = units.ANGLE
                pot_unit = units.ENERGY / units.ANGLE ** 2
            else:
                raise ValueError('The interaction_type must be "bond", "angle",'
                                 ' or "improper"')
        except KeyError as err:
            raise TypeError('An interaction_type of "bond", "angle" or improper"'
                            ' must be passed') from err
        # Create a new (uninitialized) HarmonicPotential
        h_pot = super().__new__(cls)
        # Decorate __init__ with inter_func_decorator to apply to correct units
        # (which are dependent on the interaction_type) to the equilibrium_state
        # and potential_strength parameters. This decoration has to be done in
        # new rather than directly on __init__ as it requires the dynamically
        # defined interaction_type to determine the units. cls_init (which is a
        # copy of the original cls.__init__) is required (rather than just
        # applying the decorator directly to cls.__init__) because after the
        # first HarmonicPotential is called, the cls.__init__ will be decorated.
        # Therefore, after this if cls.__init__ is decorated again, it will be
        # irrelevant, as the original decorator will be the last called. So
        # cls._init is used as it will always be equivalent to the undecorated
        # __init__ visible below.
        cls.__init__ = inter_func_decorator(eq_unit, pot_unit)(cls._init)
        return h_pot

    def __init__(self, equilibrium_state, potential_strength, **settings):

        super().__init__(locals())

    # Declare a class method equal to the __init__ method
    _init = __init__


class Periodic(InteractionFunction):

    r"""
    Periodic potential for proper and improper Dihedral interactions, with the
    form:

    .. math::

        E = \sum_{i=1,m}K_i[1.0+\textup{cos}(n_i\phi-d_i)]

    where \phi is angle between the ijk and jkl planes (where i, j, k, and l are
    the four atoms of the dihedral).

    Parameters
    ----------
    K1 : float
        The K parameter (energy) of the first order term, in units of kJ mol^-1
    n1 : int
        The n parameter of the first order term, which is unitless. This must be
        a non-negative int.
    d1 : float
        The d parameter (angle) of the first order term, in units of deg
    *params
        K, n, and d parameters for higher order terms. These must be ordered K2,
        n2, d2, K3, n3, d3, K4, n4, d4 etc. The types and units of these
        parameters are the same as the corresponding first order parameters
        listed above.
    """

    def __init__(self, K1, n1, d1, *params):

        # Check that total number of parameters is divisible by 3
        # Check that all n values are non-negative ints
        val_dict = {}
        all_params = [iter((K1, n1, d1) + params)] * 3
        for order, order_params in enumerate(zip_longest(*all_params), start=1):
            # If *params is not divisible by three, one or two indexes of
            # order_params will be None
            if None in order_params:
                raise TypeError('*params must contain a K, n, and d value for'
                                ' each order >2 (i.e. it should contain a'
                                ' number of values exactly divisible by 3)')
            if not isinstance(order_params[1], int):
                raise TypeError('All n values must be of type int')
            if order_params[1] < 0.:
                raise ValueError('All n values must be non-negative ints')
            val_dict['K{0}'.format(order)] = (units.UnitFloat(order_params[0],
                                                              units.ENERGY))
            val_dict['n{0}'.format(order)] = order_params[1]
            val_dict['d{0}'.format(order)] = (units.UnitFloat(order_params[2],
                                                              units.ANGLE))

        super().__init__(val_dict)


class LennardJones(InteractionFunction):

    r"""
    Dispersive Lennard-Jones interaction with the form:

    .. math::

        E = 4{\epsilon}[(\frac{\sigma}{r})^{12} - (\frac{\sigma}{r})^6)]
        \qquad r < r_c

    Parameters
    ----------
    epsilon : float
        The LJ epsilon value in units of kJ mol^-1
    sigma : float
        The LJ sigma value in units of Ang
    **settings
        cutoff : float
            The distance in Ang at which the potential is cutoff
        long_range_solver : str
            The long range solver, either 'PPPM', 'PME', or 'E' for
            Particle-Particle Particle-Mesh, Particle Mesh Ewald, or Ewald
            solvers
    """

    @inter_func_decorator(units.ENERGY, units.LENGTH)
    def __init__(self, epsilon, sigma, **settings):

        super().__init__(locals())
        self.cutoff = settings.get('cutoff', None)
        self.solver = settings.get('long_range_solver', None)


def filter_parameters(parameters, predicate):

    """
    Filters a list of Parameter objects using a predicate

    Parameters
    ----------
    parameters : list
        A list of Parameter objects.
    predicate : function
        A function that returns a boolean which takes a Parameter as an
        argument.

    Returns
    -------
    list
        A list of Parameter objects which meet the condition of the predicate
    """

    return list(filter(predicate, parameters))


def filter_parameters_name(parameters, name):

    """
    Filters a list of Parameters objects by name

    Parameters
    ----------
    parameters : list
        A list of Parameter objects.
    name : str
        The name of the Parameter objects to return.

    Returns
    -------
    list
        A list of Parameter objects with name
    """

    return list(filter(lambda p: p.name == name, parameters))


def filter_parameters_value(parameters, comparison, value):

    """
    Filters a list of Parameters objects by value

    Parameters
    ----------
    parameters : list
        A list of Parameter objects.
    comparison : str
        A string representing a comparison operator, '>', '<', '>=', '<=', '==',
        '!='.
    value : float
        A float with which Parameter values are compared, using the comparison
        operator.

    Returns
    -------
    list
        A list of Parameter objects which return a True when their values are
        compared with value using the comparison operator
    """

    ops = {'>':operator.gt,
           '<':operator.lt,
           '>=':operator.ge,
           '<=':operator.le,
           '==':operator.eq,
           '!=':operator.ne}

    return list(filter(lambda p: ops[comparison](p.value, value), parameters))


def filter_parameters_interaction(parameters, interaction_name):

    """
    Filters a list of Parameters objects by Interaction.name

    Parameters
    ----------
    parameters : list
        A list of Parameter objects.
    interaction_name : str
        The name of the Interaction of Parameter objects to return, for example
        'Bond'.

    Returns
    -------
    list
        A list of Parameter objects which have an interaction with the specified
        name
    """

    return list(filter(lambda p: p.interactions_name == interaction_name,
                       parameters))


def filter_parameters_function(parameters, function_name):

    """
    Filters a list of Parameters objects by InteractionFunction.name

    Parameters
    ----------
    parameters : list
        A list of Parameter objects.
    function_name : str
        The name of the InteractionFunction of Parameter objects to return, for
        example 'LennardJones' or 'HarmonicPotential'.

    Returns
    -------
    list
        A list of Parameter objects which have a function with the specified
        name
    """

    return list(filter(lambda p: p.functions_name == function_name, parameters))


def filter_parameters_atom_attribute(parameters, attribute, value):

    """
    Filters a list of Parameters objects by attribute of Atoms which have the
    Parameter applied to them


    Parameters
    ----------
    parameters : list
        A list of Parameter objects.
    attribute : str
        An attribute of an Atom. Attributes to match to must be either float or
        str.
    value : str, float
        The value of the Atom attribute.

    Returns
    -------
    list
        A list of Parameter objects which are applied to an Atom object which
        has the specified value of the specified attribute
    """

    return list(filter(lambda p: value in [getattr(atom, attribute)
                                           for int in p.interactions
                                           for atom
                                           in chain.from_iterable(int.atoms)
                                          ], parameters))


def filter_parameters_structure(parameters, structure_name):

    """
    Filters a list of Parameters objects by the name of the structural units to
    which they apply

    Parameters
    ----------
    parameters : list
        A list of Parameter objects.
    structure_name : str
        The name of a structural_unit.

    Returns
    -------
    list
        A list of Parameter objects which are applied to a structural_unit which
        has the specified name
    """

    def check_structure_name(parameter):

        """
        Checks the name of all structures

        Returns
        -------
        list
            A list of str with the of names of structural_units
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

    return list(filter(check_structure_name, parameters))
