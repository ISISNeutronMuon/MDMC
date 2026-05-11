# MDMC is a package for the optimisation of classical potentials with experimental data
# Copyright (C) 2026 MDMC Developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
        q_O = -0.84844  # e
        q_H = abs(q_O / 2)  # e

        # LJ Parameters
        sigma = 3.1780  # Ang
        epsilon = 0.65214  # kJ mol^-1

        # Bond Parameters
        r_OH = 1.0118  # Ang
        f_OH = 1882.8  # kJ mol^-1 Ang^-2

        # Bond Angle Parameters
        a_HOH = 108.15  # deg
        f_HOH = 230.12  # kJ mol^-1 rad^-2

        return {
            (Coulombic, ("O",)): Coulomb(q_O),
            (Coulombic, ("H",)): Coulomb(q_H),
            (Dispersion, ("O", "O")): LennardJones(epsilon, sigma),
            (Bond, ("H", "O")): HarmonicPotential(r_OH, f_OH, interaction_type="bond"),
            (BondAngle, ("H", "O", "H")): HarmonicPotential(a_HOH, f_HOH, interaction_type="angle"),
        }
