"""A module for defining force fields that can be applied to a universe

Each force field consists of a combination of interaction functions, and also
the values of the parameters within these functions.  In this instance water
models (such as SPCE and TIP3P) are also defined as force fields, even though
the parameter sets are restricted to describing water.

AUTHOR :    Thomas Farmer        START DATE :    2018-5-4 17:38:48"""

from abc import ABCMeta,abstractmethod

import MDMC.src.MD.interaction_functions as ifu
import MDMC.src.MD.structural_units as su

# TODO: Implement factory pattern for force fields
# TODO: Make dictionary an abstractproperty but for a class if this is possible
class ForceField:

    """
    Abstract class defining a force field

    For each interaction type that it uses (non-bonded, bonds, bond angles etc),
    a force field must define the interaction function (LJ, harmonic etc).  It
    must also define the parameters for each of these functions.
    """

    __metaclass__ = ABCMeta

    def __init__(self,interactions):
        for interaction in interactions:
            self.parameterize_interaction(interaction)

    def parameterize_interaction(self,interaction):
        int_type = type(interaction)
        elements = interaction._element_tuple()
        try:
            interaction.function = self.interaction_dictionary[
                (int_type,elements)]
        except KeyError:
            raise KeyError("This force field does not have defined interactions"
                " for these element types")


class SPCE(ForceField):

    """
    SPCE force field - LJ,Coulombic, fixed bond lengths and angles
    """

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

    interaction_dictionary = {(su.Coulombic,('O',)):ifu.Coulomb(q_O),
        (su.Coulombic,('H',)):ifu.Coulomb(q_H),
        (su.Dispersion,('O',)):ifu.LennardJones(sigma, eta),
        (su.Bond,('H','O')):ifu.HarmonicPotential(r_OH,f_OH),
        (su.BondAngle,('H','O','H')):ifu.HarmonicPotential(a_HOH,f_HOH)}

    def __init__(self,interactions):
        super(SPCE,self).__init__(interactions)


class SPC(ForceField):

    """
    SPC force field - LJ,Coulombic, fixed bond lengths and angles
    """

    # Charge Params
    q_O = -0.82         # e
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

    def __init__(self, interactions):
        super(SPC, self).__init__(interactions)

    interaction_dictionary = {(su.Coulombic,('O',)):ifu.Coulomb(q_O),
        (su.Coulombic,('H',)):ifu.Coulomb(q_H),
        (su.Dispersion,('O',)):ifu.LennardJones(sigma,eta),
        (su.Bond,('H','O')):ifu.HarmonicPotential(r_OH,f_OH),
        (su.BondAngle,('H','O','H')):ifu.HarmonicPotential(a_HOH,f_HOH)}
