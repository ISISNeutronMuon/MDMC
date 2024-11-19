"""A module for defining the TIP3P-FB forcefield

This definition of the TIP3P-FB forcefield includes bond and bond angle
strengths as these are needed for to create the required HarmonicPotentials. As
a result, they can be used for simulating a flexible water molecule. However,
TIP3P-FB itself is a rigid model, and in order to replicate this a constraint
algorithm should be used for all Bond and BondAngle objects.

Parameters (excluding bond strengths) are from:
    Building Force Fields: An Automatic, Systematic, and Reproducible Approach
    Lee-Ping  Wang,  Todd  J.  Martinez,  and  Vijay  S.  Pande
    J. Phys. Chem. Lett. 2014, 5, 11, 1885–1891

The strengths provided are the same as those used for TIP3P, from:
    https://lammps.sandia.gov/doc/Howto_tip3p.html
having converted from their units of kcal to our kJ.

Note that different values for bond strengths are given in the OPLSAA data
file, namely 2510.4 and 313.8 respectively."""

from MDMC.MD.force_fields.ff import WaterModel
from MDMC.MD.interaction_functions import Coulomb, HarmonicPotential, LennardJones
from MDMC.MD.interactions import Bond, BondAngle, Coulombic, Dispersion


class TIP3PFB(WaterModel):

    """
    TIP3P-FB force field - LJ, Coulombic, fixed bond lengths and angles
    """

    n_body = 3

    @property
    def interaction_dictionary(self):

        # Parameters from:
        # Building Force Fields: An Automatic, Systematic, and Reproducible
        # Approach
        # Lee-Ping  Wang,  Todd  J.  Martinez,  and  Vijay  S.  Pande
        # J. Phys. Chem. Lett. 2014, 5, 11, 1885–1891

        # Charge Parameters
        q_O = -0.84844      # e
        q_H = abs(q_O/2)    # e

        # LJ Parameters
        sigma = 3.1780      # Ang
        epsilon = 0.65214   # kJ mol^-1

        # Bond Parameters
        r_OH = 1.0118       # Ang
        f_OH = 1882.8       # kJ mol^-1 Ang^-2

        # Bond Angle Parameters
        a_HOH = 108.15      # deg
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
