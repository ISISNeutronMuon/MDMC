"""A module for storing atomic interaction functions

Contains abstract class InteractionFunction for defining the form interaction
function classes must take.  All functions describing atomic interactions must
be added to this module in order to be called by a universe.  If needed, the
interaction function classes can be extended to contain actual function
definitions.

AUTHOR :    Thomas Farmer        START DATE :    2018-5-1 10:15:10"""

from abc import ABCMeta, abstractmethod
from collections import MutableMapping
from inspect import getargspec, getmembers
import weakref

import numpy as np

class Parameter(object):

    """
    A force field parameter which can be fixed or constrained within limits

    The value of a parameter cannot be set if fixed=True.  The constraints
    specif

    As __repr__ returns a string of a dictionary of the public attributes (so
    that these are easy to view), for consistency this class implements some
    dictionary methods.
    """

    def __init__(self, value, name, fixed=False, constraints=None):

        """
        Attributes:
        value - float specifying the value of the parameter
        fixed - boolean specifying whether or not the value can be changed
        constraints - 2 element tuple (lower, upper) specifying the closed range
        in which value can be set
        """

        self.name = name
        self.constraints = constraints
        self.value = value
        self.fixed = fixed
        self._interactions = []

    @property
    def value(self):

        return self._value

    @value.setter
    def value(self, value):

        """
        Checks if Parameter is fixed or constrained
        """

        if hasattr(self, 'fixed') and self.fixed:
            print "Unable to change fixed parameter"
        else:
            if self.constraints is not None:
                raise NotImplementedError
            else:
                self._value = value

    @property
    def constraints(self):

        return self._constraints

    @constraints.setter
    def constraints(self, constraints):

        """
        Checks if constraints are a 2 element tuple of floats, that lower is
        less than or equal to upper, and that self.value is within them
        """

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

        self._interactions.append(weakref.ref(interaction))

    def __repr__(self):

        """
        Returns:
        The class name and a string of a dictionary containing public attributes
        including properties
        """

        # Determine which attributes are in the form of properties, and include
        # add these to public attributes in __dict__
        properties = getmembers(self.__class__,
                                lambda o: isinstance(o, property))
        prop_str = [str(p[0]) for p in properties]
        rpr = {k:v for k,v in self.__dict__.items() if '_' not in k[0]}
        for p in prop_str:
            rpr[p] = getattr(self, p)

        return '<{name} {rpr}>'.format(name=self.__class__.__name__, rpr=rpr)

    def __getitem__(self, key):

        return self.__getattribute__(key)

    def __setitem__(self, key, value):

        self.__setattr__(key, value)


class InteractionFunction(object):

    """
    Base class for interaction functions, which can be user supplied
    """

    def __init__(self, names, values):

        self.params = [Parameter(values[name], name) for name in names]

    @property
    def params(self):

        """
        Returns:
        Array of parameters
        """

        return self._params

    @params.setter
    def params(self, value):

        self._params = np.array(value)

    @property
    def name(self):

        return self.__class__.__name__

    def set_params_interactions(self, interaction):

        """
        Sets the parent interaction for all parameters
        """

        for param in self.params:

            param.interactions = interaction


class HarmonicPotential(InteractionFunction):

    """
    Harmonic potential for bond stretching and angular vibration
    """

    def __init__(self, equilibrium_state, potential_strength):

        # Get the __init__ argument list except the zeroeth index which is self
        args = getargspec(self.__class__.__init__).args[1:]
        super(self.__class__, self).__init__(args, locals())


class LennardJones(InteractionFunction):

    """
    Dispersive Lennard-Jones interaction
    """

    def __init__(self, sigma, eta):

        args = getargspec(self.__class__.__init__).args[1:]
        super(self.__class__, self).__init__(args, locals())

class Coulomb(InteractionFunction):

    """
    Coulomb interaction for charged particles
    """

    def __init__(self, charge):

        args = getargspec(self.__class__.__init__).args[1:]
        super(self.__class__, self).__init__(args, locals())
