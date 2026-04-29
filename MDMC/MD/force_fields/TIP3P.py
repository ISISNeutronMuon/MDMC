from typing import Any

from MDMC.MD.interaction_functions import HarmonicPotential, NonBonded
from MDMC.MD.interactions import Bond, BondAngle, NonBondedForce
from MDMC.MD.structures import Atom, Molecule


class TIP3PMol(Molecule):
    def __init__(self, elements=("H", "O"), constrained=True, **settings: Any):
        H1 = Atom(elements[0], position=(0.9572, 0.0, 0.0), atom_type="tip3p_H")
        H2 = Atom(elements[0], position=(-0.2400, 0.9266, 0.0), atom_type="tip3p_H")
        O = Atom(elements[1], position=(0.0, 0.0, 0.0), atom_type="tip3p_O")
        settings = {
            "position": (0, 0, 0),
            "atoms": [H1, H2, O],
            "interactions": [
                Bond((H1, O), (H2, O), constrained=constrained),
                BondAngle((H1, O, H2), constrained=constrained),
            ],
            "name": "tip3p",
        }
        super().__init__(**settings)


def add_tip3p_ff(universe, cutoff, ewald, constrained=True):
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
        function=NonBonded(charge=q_H, epsilon=0.0, sigma=0.0),
    )
    nonbonded.function.charge.parameter_name = "tip3p_H_nonbonded_charge"
    nonbonded.function.epsilon.parameter_name = "tip3p_H_nonbonded_epsilon"
    nonbonded.function.sigma.parameter_name = "tip3p_H_nonbonded_sigma"

    if not constrained:
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
        harmonicangle.potential_strength.parameter_name = (
            "tip3p_HOH_harmonicangle_potential_strength"
        )

        for interaction in universe.interactions:
            if isinstance(interaction, BondAngle):
                interaction.function = harmonicangle
