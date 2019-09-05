"""A module for defining the SPCE forcefield

This definition of the SPCE forcefield includes bond and bond angle strengths,
and so can be used for simulating a flexible SPCE water molecule

AUTHOR :    Thomas Farmer        START DATE :    02/11/2018, 13:24:21"""

from MDMC.common import units
from MDMC.common.units import UnitFloat
from MDMC.MD.force_fields.ff import WaterModel
import MDMC.MD.structural_units as su
import MDMC.MD.interaction_functions as ifu

class SPCE(WaterModel):

    """
    SPCE force field - LJ, Coulombic, fixed bond lengths and angles
    """

    n_body = 3

    @property
    def interaction_dictionary(self):

        # Parameters from:
        # O. Telemann, B. Jonsson, S. Engstrom
        # Mol. Phys. 60(1), 193-203 (1987)

        # Charge Params
        q_O = UnitFloat(-0.8476, units.CHARGE)
        q_H = UnitFloat(abs(q_O/2), units.CHARGE)

        # LJ Params
        sigma = UnitFloat(3.166, units.LENGTH)       # Ang
        epsilon = UnitFloat(0.6502, units.ENERGY / units.AMOUNT)    # kJ mol^-1

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
            (su.Dispersion, ('O', 'O')):ifu.LennardJones(epsilon, sigma),
            (su.Bond, ('H', 'O')):ifu.HarmonicPotential(r_OH, f_OH),
            (su.BondAngle, ('H', 'O', 'H')):ifu.HarmonicPotential(a_HOH, f_HOH)}
