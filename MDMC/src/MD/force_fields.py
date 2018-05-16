"""A module for defining force fields that can be applied to a universe

Each force field consists of a combination of interaction functions, and also
the values of the parameters within these functions.  In this instance water
models (such as SPCE and TIP3P) are also defined as force fields, even though
the parameter sets are restricted to describing water.

AUTHOR :    Thomas Farmer        START DATE :    2018-5-4 17:38:48"""

from abc import ABC,abstractmethod

import MDMC.src.MD.interaction_functions as ifu
import MDMC.src.MD.structural_units as su

class ForceField(ABC):
    """Abstract class defining a force field

    For each interaction type that it uses (non-bonded, bonds, bond angles etc),
    a force field must define the interaction function (LJ, harmonic etc).  It
    must also define the parameters for each of these functions.
    """

    def __init__(self,interactions):
        for interaction in interactions:
            self.parameterize_interaction(interaction)

    def parameterize_interaction(self,interaction):
        int_type = type(interaction)
        elements = interaction._element_tuple()
        interaction.function = self.interaction_dictionary()[(int_type,elements)]
        # TODO: try catch for interactions and element combinations that do not occur

    # TODO: Find a better solution than this:
    @abstractmethod
    def interaction_dictionary(self):
        return NotImplementedError

class SPCE(ForceField):
    """SPCE force field - LJ,Coulombic, fixed bond lengths and angles
    """
    # TODO: extract hard coded parameters into a seperate file

    # Parameters from:
    # O. Telemann, B. Jonsson, S. Engstrom
    # Mol. Phys. 60(1), 193-203 (1987)

    # Charge Params
    q_O = -0.8476       # e
    q_H = abs(q_O/2)    # e

    # LJ Params
    sigma = 3.166       # angstrom
    eta = 0.6502        # kJ mol^-1

    # Bond Params
    r_OH = 1.000       # angstrom
    f_OH = 4637.       #  kJ mol^-1 A^-2

    # Bond Angle Params
    a_HOH = 109.47    # deg
    f_HOH = 383.      # kJ mol^-1 rad^2

    def __init__(self,interactions):
        super().__init__(interactions)

    # TODO: Recreates interaction dictionary with each call - change this, but maintain new potential object generation
    # TODO: Replace with abstract factory
    def interaction_dictionary(self):
        return {(su.Coulombic,('O',)):ifu.coulomb(self.q_O),
                (su.Coulombic,('H',)):ifu.coulomb(self.q_H),
                (su.Dispersion,('O',)):ifu.lennard_jones(self.sigma,self.eta),
                (su.Bond,('H','O')):ifu.harmonic_potential(self.r_OH,self.f_OH),
                (su.BondAngle,('H','O','H')):ifu.harmonic_potential(self.a_HOH,self.f_HOH)}
