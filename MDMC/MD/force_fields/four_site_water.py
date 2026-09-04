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

from typing import Any

import numpy as np

from MDMC.MD.interaction_functions import DummyInteractionFunction, HarmonicPotential, NonBonded
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
        elements: tuple[str, str, str] = ("H", "O", "M"),
        model_name: str = "TIP4P",
        **settings: Any,
    ):
        if model_name not in PARAMETERS:
            raise KeyError(
                f"{model_name!r} is not available, only the following "
                f"four site water models are implemented: "
                f"{', '.join(PARAMETERS.keys())}.",
            )

        oh = PARAMETERS[model_name]["OH"]
        hoh = PARAMETERS[model_name]["HOH"]
        x = oh * np.cos(np.deg2rad(90 - hoh / 2))
        y = oh * np.sin(np.deg2rad(90 - hoh / 2))
        H1 = Atom(
            elements[0], position=(x, y, 0.0), atom_type=f"{model_name}-H", name=f"{model_name}-H"
        )
        H2 = Atom(
            elements[0], position=(-x, y, 0.0), atom_type=f"{model_name}-H", name=f"{model_name}-H"
        )
        O1 = Atom(
            elements[1],
            position=(0.0, 0.0, 0.0),
            atom_type=f"{model_name}-O",
            name=f"{model_name}-O",
        )

        om = PARAMETERS[model_name]["OM"]
        w_H = om / (2 * y)
        w_O = 1 - 2 * w_H
        M = AverageSite3P(
            elements[2],
            particles=(O1, H1, H2),
            weights=[w_O, w_H, w_H],
            atom_type=f"{model_name}-M",
            name=f"{model_name}-M",
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
    if model_name not in PARAMETERS:
        raise KeyError(
            f"{model_name!r} is not available, only the following "
            f"four site water models are implemented: "
            f"{', '.join(PARAMETERS.keys())}.",
        )

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
        function=NonBonded(
            charge=0.0, epsilon=lj_eps_O, sigma=lj_sigma_O, elements=["O"], molecules=[model_name]
        ),
    )
    nonbonded.function.charge.parameter_name = f"{model_name}-O-nonbonded_charge"
    nonbonded.function.epsilon.parameter_name = f"{model_name}-O-nonbonded_epsilon"
    nonbonded.function.sigma.parameter_name = f"{model_name}-O-nonbonded_sigma"
    universe.set_atom_charge(atom_name=f"{model_name}-O", charge=0)

    nonbonded = NonBondedForce(
        universe,
        f"{model_name}-H",
        cutoff=cutoff,
        ewald=ewald,
        function=NonBonded(
            charge=q_H, epsilon=lj_eps_H, sigma=lj_sigma_H, elements=["H"], molecules=[model_name]
        ),
    )
    nonbonded.function.charge.parameter_name = f"{model_name}-H-nonbonded_charge"
    nonbonded.function.epsilon.parameter_name = f"{model_name}-H-nonbonded_epsilon"
    nonbonded.function.sigma.parameter_name = f"{model_name}-H-nonbonded_sigma"
    universe.set_atom_charge(atom_name=f"{model_name}-H", charge=q_H)

    nonbonded = NonBondedForce(
        universe,
        f"{model_name}-M",
        cutoff=cutoff,
        ewald=ewald,
        function=NonBonded(
            charge=q_M, epsilon=0.0, sigma=1.0, elements=["M"], molecules=[model_name]
        ),
    )
    nonbonded.function.charge.parameter_name = f"{model_name}-M-nonbonded_charge"
    nonbonded.function.epsilon.parameter_name = f"{model_name}-M-nonbonded_epsilon"
    nonbonded.function.sigma.parameter_name = f"{model_name}-M-nonbonded_sigma"
    universe.set_atom_charge(atom_name=f"{model_name}-M", charge=q_M)

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
        if isinstance(interaction, Bond):
            atm_i, atm_j = interaction.atoms[0]
            atm_types = sorted((atm_i.name, atm_j.name))
            if atm_types == [f"{model_name}-H", f"{model_name}-O"]:
                interaction.function = harmonicbond
            elif atm_types == [f"{model_name}-H", f"{model_name}-M"] or atm_types == [
                f"{model_name}-O",
                f"{model_name}-M",
            ]:
                # add some dummy interaction function
                interaction.function = DummyInteractionFunction()
        elif isinstance(interaction, BondAngle):
            atm_i, atm_j, atm_k = interaction.atoms[0]
            if sorted((atm_i.name, atm_j.name, atm_k.name)) == [
                f"{model_name}-H",
                f"{model_name}-H",
                f"{model_name}-O",
            ]:
                interaction.function = harmonicangle
