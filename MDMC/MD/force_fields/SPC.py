"""A module for defining the SPC forcefield

This definition of the SPC forcefield includes bond and bond angle strengths,
and so can be used for simulating a flexible SPC water molecule

AUTHOR :    Thomas Farmer        START DATE :    02/11/2018, 13:24:21"""

from MDMC.common import units
from MDMC.common.units import UnitFloat
from MDMC.MD.force_fields.ff import WaterModel
import MDMC.MD.structural_units as su
from MDMC.MD.interaction_functions import (Coulomb, HarmonicPotential,
                                           LennardJones)


class SPC(WaterModel):

    """
    SPC force field - LJ, Coulombic, fixed bond lengths and angles
    """

    n_body = 3

    @property
    def interaction_dictionary(self):

        # Charge Params
        q_O = -0.82         # e
        q_H = abs(q_O/2)    # e

        # LJ Params
        sigma = 3.166      # Ang
        epsilon = 0.6502   # kJ mol^-1

        # Bond Params
        r_OH = 1.000       # Ang
        f_OH = 4637.       # kJ mol^-1 Ang^-2

        # Bond Angle Params
        a_HOH = 109.47     # deg
        f_HOH = 383.       # kJ mol^-1 rad^-2

        return {
            (su.Coulombic, ('O',)):Coulomb(q_O),
            (su.Coulombic, ('H',)):Coulomb(q_H),
            (su.Dispersion, ('O', 'O')):LennardJones(epsilon, sigma),
            (su.Bond,
             ('H', 'O')):HarmonicPotential(r_OH, f_OH, interaction_type='bond'),
            (su.BondAngle,
             ('H', 'O', 'H')):HarmonicPotential(a_HOH, f_HOH,
                                                interaction_type='angle')}
