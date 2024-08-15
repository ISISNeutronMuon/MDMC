"""A module for defining the SPCE forcefield

This definition of the SPCE forcefield includes bond and bond angle strengths
as these are needed for to create the required HarmonicPotentials. As a result,
they can be used for simulating a flexible water molecule. However, SPCE
itself is a rigid model, and in order to replicate this a constraint algorithm
should be used for all Bond and BondAngle objects.

Parameters (excluding bond strengths) are from:
    The missing term in effective pair potentials
    H. J. C. Berendsen, J. R. Grigera, and T. P. Straatsma
    J. Phys. Chem. 1987, 91, 24, 6269–6271

The strengths provided are the same as those used for SPC, from:
    A molecular dynamics simulation of a water model with intramolecular degrees of freedom
    O. Teleman, B. Jonsson, S. Engstrom
    Mol. Phys. 60(1), 193-203 (1987)

Note that different values for bond strengths are given in the OPLSAA data
file, namely 2510.4 and 313.8 respectively."""

from MDMC.MD.force_fields.ff import WaterModel
from MDMC.MD.interaction_functions import Coulomb, HarmonicPotential, LennardJones
from MDMC.MD.interactions import Bond, BondAngle, Coulombic, Dispersion


class SPCE(WaterModel):

    """
    SPCE force field - LJ, Coulombic, fixed bond lengths and angles
    """

    n_body = 3

    @property
    def interaction_dictionary(self):

        # Charge Parameters
        q_O = -0.8476       # e
        q_H = abs(q_O/2)    # e

        # LJ Parameters
        sigma = 3.166      # Ang
        epsilon = 0.6502   # kJ mol^-1

        # Bond Parameters
        r_OH = 1.000       # Ang
        f_OH = 4637.       # kJ mol^-1 Ang^-2

        # Bond Angle Parameters
        a_HOH = 109.47     # deg
        f_HOH = 383.       # kJ mol^-1 rad^-2

        return {
            (Coulombic, ('O',)): Coulomb(q_O),
            (Coulombic, ('H',)): Coulomb(q_H),
            (Dispersion, ('O', 'O')): LennardJones(epsilon, sigma),
            (Bond,
             ('H', 'O')): HarmonicPotential(r_OH, f_OH, interaction_type='bond'),
            (BondAngle,
             ('H', 'O', 'H')): HarmonicPotential(a_HOH, f_HOH,
                                                 interaction_type='angle')}
