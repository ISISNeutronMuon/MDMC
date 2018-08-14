"""A module for storing atomic interaction functions

Contains abstract class InteractionFunction for defining the form interaction
function classes must take.  All functions describing atomic interactions must
be added to this module in order to be called by a universe.  If needed, the
interaction function classes can be extended to contain actual function
definitions.

AUTHOR :    Thomas Farmer        START DATE :    2018-5-1 10:15:10"""

from abc import ABCMeta, abstractmethod
from collections import MutableMapping
from inspect import getmembers


class InteractionFunction:

    """
    Abstract class defining form of interaction functions, which can be user
    supplied
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):

        raise NotImplementedError


    @property
    def params(self):

        """
        Returns:
        The params key and the attributes of the Parameter associated with it.
        """

        return self._params


class Parameter(object):

    """
    A force field parameter which can be fixed or constrained within limits

    The value of a parameter cannot be set if fixed=True.  The constraints
    specif

    As __repr__ returns a string of a dictionary of the public attributes (so
    that these are easy to view), for consistency this class implements some
    dictionary methods.
    """

    def __init__(self, value, fixed=False, constraints=None):

        """
        Attributes:
        value - float specifying the value of the parameter
        fixed - boolean specifying whether or not the value can be changed
        constraints - 2 element tuple (lower, upper) specifying the closed range
        in which value can be set
        """

        self.constraints = constraints
        self.value = value
        self.fixed = fixed

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


class ParameterDict(MutableMapping):

    """
    A dictionary for storing the Parameters of an InteractionFunction
    """

    def __init__(self, *args, **kwargs):

        self.__dict__.update(*args, **kwargs)

    def __setitem__(self, key, value):

        """
        If value is a dictionary then attributes in Parameter using this
        dictionary. If value is a float (or can be cast to a float),
        Parameter.value is set to value.
        """

        if isinstance(value, dict):
            for attr, val in value.items():
                if hasattr(self.__dict__[key], attr):
                    setattr(self.__dict__[key], attr, val)
                else:
                    raise AttributeError(
                        "Parameter does not have that attribute")
        else:
            self.__dict__[key].value = float(value)

    def __getitem__(self, key):

        return self.__dict__[key]

    def __delitem__(self, key):

        del self.__dict__[key]

    def __iter__(self):

        return iter(self.__dict__)

    def __len__(self):

        return len(self.__dict__)

    def __repr__(self):

        return str(self.__dict__)


class HarmonicPotential(InteractionFunction):

    """
    Harmonic potential for bond stretching and angular vibration
    """

    def __init__(self, equilibrium_state, potential_strength):

        self._params = ParameterDict({'equilibrium_state':
                                      Parameter(equilibrium_state),
                                      'potential_strength':
                                      Parameter(potential_strength)})


class LennardJones(InteractionFunction):

    """
    Dispersive Lennard-Jones interaction
    """

    def __init__(self, sigma, eta):
        self._params = ParameterDict({'sigma':Parameter(sigma),
                                      'eta':Parameter(eta)})

class Coulomb(InteractionFunction):

    """
    Coulomb interaction for charged particles
    """

    def __init__(self, charge):
        self._params = ParameterDict({'charge':Parameter(charge)})
