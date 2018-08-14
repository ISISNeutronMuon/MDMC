"""A module for storing atomic interaction functions

Contains abstract class InteractionFunction for defining the form interaction
function classes must take.  All functions describing atomic interactions must
be added to this module in order to be called by a universe.  If needed, the
interaction function classes can be extended to contain actual function
definitions.

AUTHOR :    Thomas Farmer        START DATE :    2018-5-1 10:15:10"""

from abc import ABCMeta, abstractmethod, abstractproperty
from collections import MutableMapping


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
    A force field parameter
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
