"""A module for storing atomic interaction functions

Contains class InteractionFunction from which all interaction function classes
must derive.  All functions describing atomic interactions must be added to this
module in order to be called by a universe.  If needed, the interaction function
classes can be extended to contain actual function definitions.

Contains class Parameter, which defines the name and value of each
parameter which belongs to an InteractionFunction, and whether the parameter is
fixed, has constraints or is tied.

Contains filters for filtering list of parameters based on a predicate.

AUTHOR :    Thomas Farmer        START DATE :    2018-5-1 10:15:10"""

import ast
from inspect import getargspec, getmembers
import operator
import warnings
import weakref

import numpy as np

from MDMC.common.decorators import unit_decorator, unit_decorator_getter
from MDMC.common import units


class Parameter(object):

    """
    A force field parameter which can be fixed or constrained within limits

    The value of a parameter cannot be set if fixed=True.  The constraints
    specif

    As __repr__ returns a string of a dictionary of the public attributes (so
    that these are easy to view), for consistency this class implements some
    dictionary methods.
    """

    def __init__(self, value, name, fixed=False, constraints=None, **settings):

        """
        Arguments:
        value - float specifying the value of the parameter
        name - a string specifying the name
        fixed - boolean specifying whether or not the value can be changed
        constraints - 2 element tuple (lower, upper) specifying the closed range
        in which value can be set. Constraints must have the same units as
        value.

        Settings:
        unit - a string specifying the unit
        """

        self.name = name
        self.unit = settings['unit'] if 'unit' in settings else value.unit
        self.constraints = constraints
        self.value = value
        self.fixed = fixed
        self.interactions_name = None
        self.functions_name = None
        self._interactions = []
        self._tie = None

    @property
    def value(self):

        if self.tied:
            return self.tie
        return self._value

    @value.setter
    def value(self, value):

        """
        Checks if Parameter is fixed or constrained
        """

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

        return self._constraints

    @constraints.setter
    def constraints(self, constraints):

        """
        Checks if constraints are a 2 element tuple of floats, that the zeroeth
        element is less than or equal to the first, and that self.value is
        within them, if it exists
        """

        if constraints is not None:
            if constraints[0] > constraints[1]:
                raise ValueError("Constaints must be (lower, upper)")
            if hasattr(self, 'value'):
                self.validate_value(self.value, constraints)
        self._constraints = constraints

    @property
    def interactions(self):

        """
        Returns:
        A list of all parent Interaction objects for this Parameter object
        """

        return [interaction() for interaction in self._interactions]

    @interactions.setter
    def interactions(self, interaction):

        """
        Appends to a list of parent Interaction objects for this Parameter
        object
        """

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
        Returns:
        The Parameter object that this Parameter is tied to, or None
        """

        if self._tie is None:
            return None
        else:
            return eval(compile(self._tie, '', 'eval'))

    @property
    def tied(self):

        """
        Returns:
        True if this Parameter is tied to another Parameter, else False
        """

        if hasattr(self, 'tie') and self.tie is not None:
            return True
        else:
            return False

    def set_tie(self, parameter, expr):

        """
        This ties the parameter's value to the value of another parameter

        Arguments:
        parameter - a Parameter object
        expr - a mathematical expression

        Example:
        set_tie(p1, "* 2") means this parameter's value will return p1.value * 2
        """

        self._tie_param = weakref.ref(parameter)
        self._tie = ast.parse('self._tie_param().value' + expr, mode='eval')

    def __repr__(self):

        """
        Returns:
        The Parameter name and a dictionary containing properties and their
        values, except self.interactions and self.tie
        """

        # Determine which attributes are in the form of properties
        properties = getmembers(self.__class__,
                                lambda o: isinstance(o, property))
        excluded = ['interactions', 'tie']
        rpr = {p[0]:getattr(self, p[0]) for p in properties
               if p[0] not in excluded}

        return '{name} = {rpr}'.format(name=self.name.replace('_', ' '),
                                       rpr=rpr)

    def __getitem__(self, key):

        return self.__getattribute__(key)

    def __setitem__(self, key, value):

        self.__setattr__(key, value)

    def validate_value(self, value, constraints):

        """
        Validates the parameter value by testing if it is within
        """

        if value < constraints[0] or value > constraints[1]:
            raise ValueError("Value must be within constraints")


class InteractionFunction(object):

    """
    Base class for interaction functions, which can be user supplied
    """

    def __init__(self, val_dict):

        """
        Arguments:
        names - a list of names of the parameters of the interaction function
        val_dict - a dictionary of name:value pairs. Currently this must be
        ordered alphabetically.
        """

        self.params = [Parameter(value, name) for name, value
                       in val_dict.items() if name != 'self']

    @property
    def params(self):

        """
        Returns:
        Array of parameters
        """

        return self._params

    @params.setter
    def params(self, value):

        self._params = np.array(sorted(value, key=lambda p:p.name))

    @property
    def params_values(self):

        """
        Returns:
        Array of values for all parameters
        """

        return np.array([p.value for p in self.params])

    @property
    def name(self):

        return self.__class__.__name__

    def set_params_interactions(self, interaction):

        """
        Sets the parent interaction for all parameters

        Arguments:
        interaction - an interaction
        """

        for param in self.params:

            param.interactions = interaction


class HarmonicPotential(InteractionFunction):

    """
    Harmonic potential for bond stretching and angular vibration
    """

    def __init__(self, equilibrium_state, potential_strength):

        """
        Arguments:
        equilibrium_state - in units of Ang (linear) or deg (angular)
        potential_strength - in units of kJ mol^-1 Ang^-2 (linear) or
        kJ mol^-1 rad^-2 (angular)
        """

        super(self.__class__, self).__init__(locals())


class LennardJones(InteractionFunction):

    """
    Dispersive Lennard-Jones interaction
    """

    def __init__(self, epsilon, sigma):

        """
        Arguments:
        epsilon - in units of kJ mol^-1
        sigma - in units of Ang
        """

        super(self.__class__, self).__init__(locals())


class Coulomb(InteractionFunction):

    """
    Coulomb interaction for charged particles
    """

    def __init__(self, charge):

        """
        Arguments:
        charge - in units of e
        """

        super(self.__class__, self).__init__(locals())


def filter_parameters(parameters, predicate):

    """
    Arguments:
    parameters - a list of parameters
    predicate - a function that returns a boolean

    Returns:
    a list of parameters which meet the condition of predicate
    """

    return filter(predicate, parameters)


def filter_parameters_name(parameters, name):

    """
    Arguments:
    parameters - a list of parameters
    name - a string specifying the parameter name, for example 'charge' for a
    Coulomb interaction or 'sigma' for an LJ interaction

    Returns:
    a list of parameters which meet the condition of parameter.name == name
    """

    return filter(lambda p: p.name == name, parameters)


def filter_parameters_value(parameters, comparison, value):

    """
    Arguments:
    parameters - a list of parameters
    comparison - a string representing a comparison operator: '>', '<', '>=',
    '<=', '==', '!='

    Returns:
    a list of parameters which meet the condition of
    parameter.value comparison value e.g. parameter.value > value
    """

    ops = {'>':operator.gt,
           '<':operator.lt,
           '>=':operator.ge,
           '<=':operator.le,
           '==':operator.eq,
           '!=':operator.ne}

    return filter(lambda p: ops[comparison](p.value, value), parameters)


def filter_parameters_interaction(parameters, interaction_name):

    """
    Arguments:
    parameters - a list of parameters
    interaction_name - a string specifying the interaction name, for example
    'Bond' for a bonded interaction

    Returns:
    a list of parameters where the interaction meets the condition of
    interaction.name == interaction_name
    """

    return filter(lambda p: p.interactions_name == interaction_name, parameters)


def filter_parameters_function(parameters, function_name):

    """
    Arguments:
    parameters - a list of parameters
    function_name - a string specifying the interaction function name, for
    example 'LennardJones' or 'HarmonicPotential'

    Returns:
    a list of parameters where the interaction function meets the condition of
    function.name == function_name
    """

    return filter(lambda p: p.functions_name == function_name, parameters)


def filter_parameters_atom_attribute(parameters, attribute, value):

    """
    Arguments:
    parameters - a list of parameters
    attribute - a string specifying an attribute of an Atom.  Attributes are
    restricted to float or strings.
    value - the desired value of the attribute

    Returns:
    a list of parameters which relate to any atom where atom.attribute == value
    """

    return filter(lambda p: value in [getattr(atom, attribute)
                                      for int in p.interactions
                                      for atom in int.atom_list], parameters)


def filter_parameters_structure(parameters, structure_name):

    """
    Arguments:
    parameters - a list of parameters
    structure_name - a string specifying the name of a structure e.g. 'water'
    for a water molecule

    Returns:
    a list of parameters which relate to any structure where
    structure.name == structure_name
    """

    def check_structure_name(parameter):

        # Recursively add structure.name to structure_names set until the
        # structure is the top level structure
        structure_names = set()
        def add_name(structure):
            structure_names.add(structure.name)
            if structure.top_level_structure() == structure:
                return
            add_name(structure.parent)

        for int in parameter.interactions:
            for atom in int.atom_list:
                add_name(atom)
        return structure_name in structure_names

    return filter(check_structure_name, parameters)
