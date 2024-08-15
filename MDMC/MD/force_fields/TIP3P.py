"""A module for defining the TIP3P forcefield

This definition of the TIP3P forcefield includes bond and bond angle strengths
as these are needed for to create the required HarmonicPotentials. As a result,
they can be used for simulating a flexible water molecule. However, TIP3P
itself is a rigid model, and in order to replicate this a constraint algorithm
should be used for all Bond and BondAngle objects.

Parameters (excluding bond strengths) are from:
    Comparison of simple potential functions for simulating liquid water
    Jorgensen WL, Chandrasekhar J, Madura JD, Impey RW, Klein ML
    The Journal of Chemical Physics. 79 (2): 926–935 (1983)

The strengths provided are from:
    https://lammps.sandia.gov/doc/Howto_tip3p.html
having converted from their units of kcal to our kJ.

Note that different values for bond strengths are given in the OPLSAA data
file, namely 2510.4 and 313.8 respectively."""

from MDMC.MD.force_fields.ff import WaterModel
from MDMC.MD.interaction_functions import Coulomb, HarmonicPotential, LennardJones
from MDMC.MD.interactions import Bond, BondAngle, Coulombic, Dispersion


class TIP3P(WaterModel):

    """
    TIP3P force field - LJ, Coulombic, fixed bond lengths and angles
    """

    n_body = 3

    @property
    def interaction_dictionary(self):

        # Charge Parameters
        q_O = -0.834        # e
        q_H = abs(q_O/2)    # e

        # LJ Parameters
        sigma = 3.151       # Ang
        epsilon = 0.6363    # kJ mol^-1

        # Bond Parameters
        r_OH = 0.9572       # Ang
        f_OH = 1882.8       # kJ mol^-1 Ang^-2

        # Bond Angle Parameters
        a_HOH = 104.52      # deg
        f_HOH = 230.12      # kJ mol^-1 rad^-2

        return {
            (Coulombic, ('O',)): Coulomb(q_O),
            (Coulombic, ('H',)): Coulomb(q_H),
            (Dispersion, ('O', 'O')): LennardJones(epsilon, sigma),
            (Bond,
             ('H', 'O')): HarmonicPotential(r_OH, f_OH, interaction_type='bond'),
            (BondAngle,
             ('H', 'O', 'H')): HarmonicPotential(a_HOH, f_HOH,
                                                 interaction_type='angle')}
