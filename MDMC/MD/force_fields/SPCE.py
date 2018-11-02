"""A module for defining the SPCE forcefield

This definition of the SPCE forcefield includes bond and bond angle strengths,
and so can be used for simulating a flexible SPCE water molecule

AUTHOR :    Thomas Farmer        START DATE :    02/11/2018, 13:24:21"""

from MDMC.MD.force_fields.ff import ForceField
import MDMC.MD.structural_units as su
import MDMC.MD.interaction_functions as ifu

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

    interaction_dictionary = {
        (su.Coulombic, ('O',)):ifu.Coulomb(q_O),
        (su.Coulombic, ('H',)):ifu.Coulomb(q_H),
        (su.Dispersion, ('O',)):ifu.LennardJones(eta, sigma),
        (su.Bond, ('H', 'O')):ifu.HarmonicPotential(r_OH, f_OH),
        (su.BondAngle, ('H', 'O', 'H')):ifu.HarmonicPotential(a_HOH, f_HOH)}

    def __init__(self, interactions):
        super(SPCE, self).__init__(interactions)
