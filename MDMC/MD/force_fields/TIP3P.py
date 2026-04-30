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

from typing import Any

from MDMC.MD.force_fields.ff import WaterModel
from MDMC.MD.interaction_functions import Coulomb, HarmonicPotential, LennardJones, NonBonded
from MDMC.MD.interactions import Bond, BondAngle, Coulombic, Dispersion, NonBondedForce
from MDMC.MD.structures import Atom, Molecule


class TIP3P(WaterModel):
    """
    TIP3P force field - LJ, Coulombic, fixed bond lengths and angles
    """

    n_body = 3

    @property
    def interaction_dictionary(self):

        # Charge Parameters
        q_O = -0.834  # e
        q_H = abs(q_O / 2)  # e

        # LJ Parameters
        sigma = 3.151  # Ang
        epsilon = 0.6363  # kJ mol^-1

        # Bond Parameters
        r_OH = 0.9572  # Ang
        f_OH = 1882.8  # kJ mol^-1 Ang^-2

        # Bond Angle Parameters
        a_HOH = 104.52  # deg
        f_HOH = 230.12  # kJ mol^-1 rad^-2

        return {
            (Coulombic, ("O",)): Coulomb(q_O),
            (Coulombic, ("H",)): Coulomb(q_H),
            (Dispersion, ("O", "O")): LennardJones(epsilon, sigma),
            (Bond, ("H", "O")): HarmonicPotential(r_OH, f_OH, interaction_type="bond"),
            (BondAngle, ("H", "O", "H")): HarmonicPotential(a_HOH, f_HOH, interaction_type="angle"),
        }


class TIP3PMol(Molecule):
    def __init__(self, elements=("H", "O"), constrained=True, **settings: Any):
        H1 = Atom(elements[0], position=(0.9572, 0.0, 0.0), atom_type="tip3p_H")
        H2 = Atom(elements[0], position=(-0.2400, 0.9266, 0.0), atom_type="tip3p_H")
        O1 = Atom(elements[1], position=(0.0, 0.0, 0.0), atom_type="tip3p_O")
        settings = {
            "position": (0, 0, 0),
            "atoms": [H1, H2, O1],
            "interactions": [
                Bond((H1, O1), (H2, O1), constrained=constrained),
                BondAngle((H1, O1, H2), constrained=constrained),
            ],
            "name": "tip3p",
        }
        super().__init__(**settings)


def add_tip3p_ff(universe, cutoff, ewald):
    # Charge Parameters
    q_O = -0.834  # e
    q_H = abs(q_O / 2)  # e

    # LJ Parameters
    sigma = 3.151  # Ang
    epsilon = 0.6363  # kJ mol^-1

    # Bond Parameters
    r_OH = 0.9572  # Ang
    f_OH = 1882.8  # kJ mol^-1 Ang^-2

    # Bond Angle Parameters
    a_HOH = 104.52  # deg
    f_HOH = 230.12  # kJ mol^-1 rad^-2

    nonbonded = NonBondedForce(
        universe,
        "tip3p_O",
        cutoff=cutoff,
        ewald=ewald,
        function=NonBonded(charge=q_O, epsilon=epsilon, sigma=sigma),
    )
    nonbonded.function.charge.parameter_name = "tip3p_O_nonbonded_charge"
    nonbonded.function.epsilon.parameter_name = "tip3p_O_nonbonded_epsilon"
    nonbonded.function.sigma.parameter_name = "tip3p_O_nonbonded_sigma"

    nonbonded = NonBondedForce(
        universe,
        "tip3p_H",
        cutoff=cutoff,
        ewald=ewald,
        function=NonBonded(charge=q_H, epsilon=0.0, sigma=1.0),
    )
    nonbonded.function.charge.parameter_name = "tip3p_H_nonbonded_charge"
    nonbonded.function.epsilon.parameter_name = "tip3p_H_nonbonded_epsilon"
    nonbonded.function.sigma.parameter_name = "tip3p_H_nonbonded_sigma"

    harmonicbond = HarmonicPotential(
        equilibrium_state=r_OH,
        potential_strength=f_OH,
        interaction_type="bond",
    )
    harmonicbond.equilibrium_state.parameter_name = "tip3p_OH_harmonicbond_equilibrium_state"
    harmonicbond.potential_strength.parameter_name = "tip3p_OH_harmonicbond_potential_strength"
    for interaction in universe.interactions:
        if isinstance(interaction, Bond):
            interaction.function = harmonicbond

    harmonicangle = HarmonicPotential(
        equilibrium_state=a_HOH,
        potential_strength=f_HOH,
        interaction_type="angle",
    )
    harmonicangle.equilibrium_state.parameter_name = "tip3p_HOH_harmonicangle_equilibrium_state"
    harmonicangle.potential_strength.parameter_name = "tip3p_HOH_harmonicangle_potential_strength"
    for interaction in universe.interactions:
        if isinstance(interaction, BondAngle):
            interaction.function = harmonicangle
