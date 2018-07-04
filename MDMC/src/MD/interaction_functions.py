"""A module for storing atomic interaction functions

Contains abstract class InteractionFunction for defining the form interaction
function classes must take.  All functions describing atomic interactions must
be added to this module in order to be called by a universe.  If needed, the
interaction function classes can be extended to contain actual function
definitions.

AUTHOR :    Thomas Farmer        START DATE :    2018-5-1 10:15:10"""

from abc import ABCMeta, abstractmethod, abstractproperty


class InteractionFunction:

    """
    Abstract class defining form of interaction functions, which can be user
    supplied
    """

    __metaclass__ = ABCMeta


    @property
    def params(self):

        """
        Returns:
        The params key and the numerical value associated with the key, which is
        contained within a Parameter object.
        """

        return {kv[0]: kv[1].value for kv in self._params.items()}

    # TODO: If functional forms are implemented then function should be abstract
    def function(self):

        pass


class Parameter(object):

    def __init__(self, value):
        self.value = value
        self.constrained = True


class HarmonicPotential(InteractionFunction):

    """
    Harmonic potential for bond stretching and angular vibration
    """

    def __init__(self, equilibrium_state, potential_strength):

        self._params = {'equilibrium_state':Parameter(equilibrium_state),
                    'potential_strength':Parameter(potential_strength)}


class LennardJones(InteractionFunction):

    """
    Dispersive Lennard-Jones interaction
    """

    def __init__(self, sigma, eta):
        self._params = {'sigma':Parameter(sigma),'eta':Parameter(eta)}

class Coulomb(InteractionFunction):

    """
    Coulomb interaction for charged particles
    """

    def __init__(self, charge):
        self._params = {'charge':Parameter(charge)}
