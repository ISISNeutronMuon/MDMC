"""A module for defining and storing atomic interaction functions

Contains abstract class InteractionFunction for defining the form interaction
function classes must take.  All functions describing atomic interactions must
be added to this module in order to be called by a universe.

AUTHOR :    Thomas Farmer        START DATE :    2018-5-1 10:15:10"""

from abc import ABC, abstractmethod

class interaction_function(ABC):
    """Abstract class defining form of interaction functions, which can be user
    supplied

    Attributes:
    n_params - number of parameters"""

class harmonic_potential(interaction_function):
    """Harmonic potential for bond stretching and angular vibration"""

class lennard_jones(interaction_function):
    """Non-bonded Lennard-Jones interaction"""

class coulomb(interaction_function):
    """Non-bonded Coulomb interaction for charged particles"""
