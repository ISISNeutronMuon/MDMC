"""A module for defining the SPC forcefield

This definition of the SPC forcefield includes bond and bond angle strengths,
and so can be used for simulating a flexible SPC water molecule

AUTHOR :    Thomas Farmer        START DATE :    02/11/2018, 13:24:21"""

from MDMC.common import units
from MDMC.common.units import UnitFloat
from MDMC.MD.force_fields.ff import ForceField
import MDMC.MD.structural_units as su
import MDMC.MD.interaction_functions as ifu


class SPC(ForceField):

    """
    SPC force field - LJ, Coulombic, fixed bond lengths and angles
    """

    @property
    def interaction_dictionary(self):

        # Charge Params
        q_O = UnitFloat(-0.82, units.CHARGE)         # e
        q_H = UnitFloat(abs(q_O/2), units.CHARGE)    # e

        # LJ Params
        sigma = UnitFloat(3.166, units.LENGTH)       # Ang
        epsilon = UnitFloat(0.6502, units.LENGTH)    # kJ mol^-1

        # Bond Params
        r_OH = UnitFloat(1.000, units.LENGTH)        # Ang
        f_OH = UnitFloat(4637.,                      # kJ mol^-1 Ang^-2
                         units.ENERGY / units.AMOUNT * units.LENGTH**2)

        # Bond Angle Params
        a_HOH = UnitFloat(109.47, units.ANGLE)      # deg
        f_HOH = UnitFloat(383.,                     # kJ mol^-1 rad^-2
                          units.ENERGY / units.AMOUNT * units.ANGLE**2)

        return {
            (su.Coulombic, ('O',)):ifu.Coulomb(q_O),
            (su.Coulombic, ('H',)):ifu.Coulomb(q_H),
            (su.Dispersion, ('O',)):ifu.LennardJones(epsilon, sigma),
            (su.Bond, ('H', 'O')):ifu.HarmonicPotential(r_OH, f_OH),
            (su.BondAngle, ('H', 'O', 'H')):ifu.HarmonicPotential(a_HOH, f_HOH)}
