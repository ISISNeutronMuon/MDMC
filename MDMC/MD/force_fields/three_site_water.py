from typing import Any

import numpy as np

from MDMC.MD.interaction_functions import HarmonicPotential, NonBonded
from MDMC.MD.interactions import Bond, BondAngle, NonBondedForce
from MDMC.MD.structures import Atom, Molecule

PARAMETERS = {
    "TIP3P": {
        "q_O": -0.834,
        "q_H": 0.417,
        "lj_eps_O": 0.1521 * 4.184,
        "lj_sigma_O": 3.1507,
        "lj_eps_H": 0.0,
        "lj_sigma_H": 1.0,
        "K_OH": 450 * 4.184,
        "OH": 0.9572,
        "K_HOH": 55.0 * 4.184,
        "HOH": 104.52,
    },
    "TIP3P-Ewald": {
        "q_O": -0.834,
        "q_H": 0.417,
        "lj_eps_O": 0.1020 * 4.184,
        "lj_sigma_O": 3.188,
        "lj_eps_H": 0.0,
        "lj_sigma_H": 1.0,
        "K_OH": 450 * 4.184,
        "OH": 0.9572,
        "K_HOH": 55.0 * 4.184,
        "HOH": 104.52,
    },
}


class ThreeSiteWater(Molecule):
    """A three site water molecule.

    Parameters
    ----------
    elements : tuple[str, str]
        Tuple of the elements to substitute into the water model.
    constrained : bool
        Constrains the bond length and angles if true.
    model_name : str
        The name of the model of that parameters that will be taken from.
    **settings : Any
        Other setting used to create the molecule.
    """

    def __init__(
        self,
        elements: tuple[str, str] = ("H", "O"),
        constrained: bool = True,
        model_name: str = "TIP3P",
        **settings: Any,
    ):
        oh = PARAMETERS[model_name]["OH"]
        hoh = PARAMETERS[model_name]["HOH"]
        x = oh * np.cos(np.deg2rad(90 - hoh / 2))
        y = oh * np.sin(np.deg2rad(90 - hoh / 2))
        H1 = Atom(elements[0], position=(x, y, 0.0), atom_type=f"{model_name}-H")
        H2 = Atom(elements[0], position=(-x, y, 0.0), atom_type=f"{model_name}-H")
        O1 = Atom(elements[1], position=(0.0, 0.0, 0.0), atom_type=f"{model_name}-O")
        settings = {
            "position": (0, 0, 0),
            "atoms": [H1, H2, O1],
            "interactions": [
                Bond((H1, O1), (H2, O1), constrained=constrained),
                BondAngle((H1, O1, H2), constrained=constrained),
            ],
            "name": model_name,
        }
        super().__init__(**settings)


def add_three_site_water_ff(universe, cutoff: float, ewald: float, model_name: str = "TIP3P"):
    """Add a three site water model force fields to the universe assuming
    that atoms with the correct atom type exists.

    Parameters
    ----------
    universe : Universe
        The MDMC universe object.
    cutoff : float
        The cutoff distance (angstrom) used for nonbonded interactions.
    ewald : float
        The error tolerance for Ewald summation.
    model_name : str
        The name of the model of that parameters that will be taken from.
    """
    # Charge Parameters
    q_O = PARAMETERS[model_name]["q_O"]
    q_H = PARAMETERS[model_name]["q_H"]

    # LJ Parameters
    lj_eps_O = PARAMETERS[model_name]["lj_eps_O"]
    lj_sigma_O = PARAMETERS[model_name]["lj_sigma_O"]
    lj_eps_H = PARAMETERS[model_name]["lj_eps_H"]
    lj_sigma_H = PARAMETERS[model_name]["lj_sigma_H"]

    # Bond Parameters
    r_OH = PARAMETERS[model_name]["OH"]
    k_OH = PARAMETERS[model_name]["K_OH"]

    # Bond Angle Parameters
    a_HOH = PARAMETERS[model_name]["HOH"]
    k_HOH = PARAMETERS[model_name]["K_HOH"]

    nonbonded = NonBondedForce(
        universe,
        f"{model_name}-O",
        cutoff=cutoff,
        ewald=ewald,
        function=NonBonded(charge=q_O, epsilon=lj_eps_O, sigma=lj_sigma_O),
    )
    nonbonded.function.charge.parameter_name = f"{model_name}-O-nonbonded_charge"
    nonbonded.function.epsilon.parameter_name = f"{model_name}-O-nonbonded_epsilon"
    nonbonded.function.sigma.parameter_name = f"{model_name}-O-nonbonded_sigma"

    nonbonded = NonBondedForce(
        universe,
        f"{model_name}-H",
        cutoff=cutoff,
        ewald=ewald,
        function=NonBonded(charge=q_H, epsilon=lj_eps_H, sigma=lj_sigma_H),
    )
    nonbonded.function.charge.parameter_name = f"{model_name}-H-nonbonded_charge"
    nonbonded.function.epsilon.parameter_name = f"{model_name}-H-nonbonded_epsilon"
    nonbonded.function.sigma.parameter_name = f"{model_name}-H-nonbonded_sigma"

    harmonicbond = HarmonicPotential(
        equilibrium_state=r_OH,
        potential_strength=k_OH,
        interaction_type="bond",
    )
    harmonicbond.equilibrium_state.parameter_name = (
        f"{model_name}-OH-harmonicbond_equilibrium_state"
    )
    harmonicbond.potential_strength.parameter_name = (
        f"{model_name}-OH-harmonicbond_potential_strength"
    )
    for interaction in universe.interactions:
        if not isinstance(interaction, Bond):
            continue
        for atm_i, atm_j in interaction.atoms:
            if sorted((atm_i.atom_type, atm_j.atom_type)) == [f"{model_name}-H", f"{model_name}-O"]:
                interaction.function = harmonicbond

    harmonicangle = HarmonicPotential(
        equilibrium_state=a_HOH,
        potential_strength=k_HOH,
        interaction_type="angle",
    )
    harmonicangle.equilibrium_state.parameter_name = (
        f"{model_name}-HOH-harmonicangle_equilibrium_state"
    )
    harmonicangle.potential_strength.parameter_name = (
        f"{model_name}-HOH-harmonicangle_potential_strength"
    )
    for interaction in universe.interactions:
        if not isinstance(interaction, BondAngle):
            continue
        for atm_i, atm_j, atm_k in interaction.atoms:
            if sorted((atm_i.atom_type, atm_j.atom_type, atm_k.atom_type)) == [
                f"{model_name}-H",
                f"{model_name}-H",
                f"{model_name}-O",
            ]:
                interaction.function = harmonicangle
