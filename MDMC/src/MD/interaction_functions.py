"""A module for storing atomic interaction functions

Contains abstract class InteractionFunction for defining the form interaction
function classes must take.  All functions describing atomic interactions must
be added to this module in order to be called by a universe.  If needed, the
interaction function classes can be extended to contain actual function
definitions.

AUTHOR :    Thomas Farmer        START DATE :    2018-5-1 10:15:10"""

from abc import ABC, abstractmethod

class interaction_function(ABC):
    """Abstract class defining form of interaction functions, which can be user
    supplied

    Attributes:
    """

    def function(self):
        pass


class harmonic_potential(interaction_function):
    """Harmonic potential for bond stretching and angular vibration"""

    def __init__(self,equilibrium_state,potential_strength):
        self.params = {'equilibrium_state':equilibrium_state,
                    'potential_strength':potential_strength}

class lennard_jones(interaction_function):
    """Dispersive Lennard-Jones interaction"""

    def __init__(self,sigma,eta):
        self.params = {'sigma':sigma,'eta':eta}

class coulomb(interaction_function):
    """Coulomb interaction for charged particles"""

    def __init__(self,charge):
        self.params = {'charge':charge}
