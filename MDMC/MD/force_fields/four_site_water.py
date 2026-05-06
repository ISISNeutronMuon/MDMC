from typing import Any

import numpy as np

from MDMC.MD.interaction_functions import HarmonicPotential, NonBonded
from MDMC.MD.interactions import Bond, BondAngle, NonBondedForce
from MDMC.MD.structures import Atom, AverageSite3P, Molecule

PARAMETERS = {
    "TIP4P": {
        "q_M": -1.040,
        "q_H": 0.520,
        "lj_eps_O": 0.1550 * 4.184,
        "lj_sigma_O": 3.1536,
        "lj_eps_H": 0.0,
        "lj_sigma_H": 1.0,
        "OH": 0.9572,
        "HOH": 104.52,
        "OM": 0.15,
    },
    "TIP4P/Ice": {
        "q_M": -1.1794,
        "q_H": 0.5897,
        "lj_eps_O": 0.21084 * 4.184,
        "lj_sigma_O": 3.1668,
        "lj_eps_H": 0.0,
        "lj_sigma_H": 1.0,
        "OH": 0.9572,
        "HOH": 104.52,
        "OM": 0.1577,
    },
    "TIP4P/2005": {
        "q_M": -1.1128,
        "q_H": 0.5564,
        "lj_eps_O": 0.1852 * 4.184,
        "lj_sigma_O": 3.1589,
        "lj_eps_H": 0.0,
        "lj_sigma_H": 1.0,
        "OH": 0.9572,
        "HOH": 104.52,
        "OM": 0.1546,
    },
    "TIP4P-Ewald": {
        "q_M": -1.04844,
        "q_H": 0.52422,
        "lj_eps_O": 0.16275 * 4.184,
        "lj_sigma_O": 3.16435,
        "lj_eps_H": 0.0,
        "lj_sigma_H": 1.0,
        "OH": 0.9572,
        "HOH": 104.52,
        "OM": 0.1250,
    },
    "OPC": {
        "q_M": -1.3582,
        "q_H": 0.6791,
        "lj_eps_O": 0.21280 * 4.184,
        "lj_sigma_O": 3.1660,
        "lj_eps_H": 0.0,
        "lj_sigma_H": 1.0,
        "OH": 0.8724,
        "HOH": 103.60,
        "OM": 0.1594,
    },
}


class FourSiteWater(Molecule):
    """A four site water molecule.

    Parameters
    ----------
    elements : tuple[str, str, str]
        Tuple of the elements to substitute into the water model.
    model_name : str
        The name of the model of that parameters that will be taken from.
    **settings : Any
        Other setting used to create the molecule.
    """

    def __init__(
        self,
        elements: tuple[str, str, str] = ("H", "O", "X"),
        model_name: str = "TIP4P",
        **settings: Any,
    ):
        oh = PARAMETERS[model_name]["OH"]
        hoh = PARAMETERS[model_name]["HOH"]
        x = oh * np.cos(np.deg2rad(90 - hoh / 2))
        y = oh * np.sin(np.deg2rad(90 - hoh / 2))
        H1 = Atom(elements[0], position=(x, y, 0.0), atom_type=f"{model_name}-H")
        H2 = Atom(elements[0], position=(-x, y, 0.0), atom_type=f"{model_name}-H")
        O1 = Atom(elements[1], position=(0.0, 0.0, 0.0), atom_type=f"{model_name}-O")

        om = PARAMETERS[model_name]["OM"] / 2
        weight = [1, om / y, om / y]
        sum_w = sum(weight)
        M = AverageSite3P(
            particles=(O1, H1, H2),
            weights=[i / sum_w for i in weight],
            atom_type=f"{model_name}-M",
        )
        settings = {
            "position": (0, 0, 0),
            "atoms": [H1, H2, O1, M],
            "interactions": [
                Bond((H1, O1), (H2, O1), constrained=True),
                Bond((H1, M), (H2, M), (O1, M)),
                BondAngle((H1, O1, H2), constrained=True),
            ],
            "name": model_name,
        }
        super().__init__(**settings)


def add_four_site_water_ff(universe, cutoff: float, ewald: float, model_name: str = "TIP4P"):
    """Add a four site water model force fields to the universe assuming
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
    q_M = PARAMETERS[model_name]["q_M"]
    q_H = PARAMETERS[model_name]["q_H"]

    # LJ Parameters
    lj_eps_O = PARAMETERS[model_name]["lj_eps_O"]
    lj_sigma_O = PARAMETERS[model_name]["lj_sigma_O"]
    lj_eps_H = PARAMETERS[model_name]["lj_eps_H"]
    lj_sigma_H = PARAMETERS[model_name]["lj_sigma_H"]

    # Bond Parameters
    r_OH = PARAMETERS[model_name]["OH"]

    # Bond Angle Parameters
    a_HOH = PARAMETERS[model_name]["HOH"]

    nonbonded = NonBondedForce(
        universe,
        f"{model_name}-O",
        cutoff=cutoff,
        ewald=ewald,
        function=NonBonded(charge=0.0, epsilon=lj_eps_O, sigma=lj_sigma_O),
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

    nonbonded = NonBondedForce(
        universe,
        f"{model_name}-M",
        cutoff=cutoff,
        ewald=ewald,
        function=NonBonded(charge=q_M, epsilon=0.0, sigma=1.0),
    )
    nonbonded.function.charge.parameter_name = f"{model_name}-M-nonbonded_charge"
    nonbonded.function.epsilon.parameter_name = f"{model_name}-M-nonbonded_epsilon"
    nonbonded.function.sigma.parameter_name = f"{model_name}-M-nonbonded_sigma"

    harmonicbond = HarmonicPotential(
        equilibrium_state=r_OH,
        potential_strength=1.0,
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
            atm_types = sorted((atm_i.atom_type, atm_j.atom_type))
            if atm_types == [f"{model_name}-H", f"{model_name}-O"]:
                interaction.function = harmonicbond
            elif atm_types == [f"{model_name}-H", f"{model_name}-M"] or atm_types == [
                f"{model_name}-O",
                f"{model_name}-M",
            ]:
                interaction.function = None

    harmonicangle = HarmonicPotential(
        equilibrium_state=a_HOH,
        potential_strength=1.0,
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
