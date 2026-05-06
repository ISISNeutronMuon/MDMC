from typing import Any

from MDMC.MD.interaction_functions import HarmonicPotential, NonBonded
from MDMC.MD.interactions import Bond, BondAngle, NonBondedForce
from MDMC.MD.structures import Atom, Molecule, AverageSite3P


class TIP4PMol(Molecule):
    """A TIP4P2005 water molecule.

    Parameters
    ----------
    elements : tuple[str, str, str]
        Tuple of the elements to substitute into the water model.
    **settings : Any
        Other setting used to create the molecule.
    """

    def __init__(
        self,
        elements: tuple[str, str, str] = ("H", "O", "X"),
        **settings: Any,
    ):
        H1 = Atom(elements[0], position=(0.75695, 0.58588, 0.0), atom_type="tip4p_H")
        H2 = Atom(elements[0], position=(-0.75695, 0.58588, 0.0), atom_type="tip4p_H")
        O1 = Atom(elements[1], position=(0.0, 0.0, 0.0), atom_type="tip4p_O")
        M = AverageSite3P(particles=(O1, H1, H2), weights=[0.79122, 0.10439, 0.10439], atom_type="tip4p_M")
        settings = {
            "position": (0, 0, 0),
            "atoms": [H1, H2, O1, M],
            "interactions": [
                Bond((H1, O1), (H2, O1), constrained=True),
                Bond((H1, M), (H2, M), (O1, M), constrained=True),
                BondAngle((H1, O1, H2), constrained=True),
            ],
            "name": "tip4p",
        }
        super().__init__(**settings)


def add_tip4p2005_ff(universe, cutoff: float, ewald: float):
    """Add the tip4p/2005 force field to the universe assuming that atoms
    with the tip4p atom type exists.

    Parameters
    ----------
    universe : Universe
        The MDMC universe object.
    cutoff : float
        The cutoff distance (angstrom) used for nonbonded interactions.
    ewald : float
        The error tolerance for Ewald summation.

    """
    # Charge Parameters
    q_M = -1.1128 # e
    q_H = abs(q_M / 2)  # e

    # LJ Parameters
    sigma = 3.1589  # Ang
    epsilon = 0.7748768  # kJ mol^-1

    # Bond Parameters
    r_OH = 0.9572  # Ang

    # Bond Angle Parameters
    a_HOH = 104.52  # deg

    nonbonded = NonBondedForce(
        universe,
        "tip4p_O",
        cutoff=cutoff,
        ewald=ewald,
        function=NonBonded(charge=0.0, epsilon=epsilon, sigma=sigma),
    )
    nonbonded.function.charge.parameter_name = "tip4p_O_nonbonded_charge"
    nonbonded.function.epsilon.parameter_name = "tip4p_O_nonbonded_epsilon"
    nonbonded.function.sigma.parameter_name = "tip4p_O_nonbonded_sigma"

    nonbonded = NonBondedForce(
        universe,
        "tip4p_H",
        cutoff=cutoff,
        ewald=ewald,
        function=NonBonded(charge=q_H, epsilon=0.0, sigma=1.0),
    )
    nonbonded.function.charge.parameter_name = "tip4p_H_nonbonded_charge"
    nonbonded.function.epsilon.parameter_name = "tip4p_H_nonbonded_epsilon"
    nonbonded.function.sigma.parameter_name = "tip4p_H_nonbonded_sigma"

    nonbonded = NonBondedForce(
        universe,
        "tip4p_M",
        cutoff=cutoff,
        ewald=ewald,
        function=NonBonded(charge=q_M, epsilon=0.0, sigma=1.0),
    )
    nonbonded.function.charge.parameter_name = "tip4p_M_nonbonded_charge"
    nonbonded.function.epsilon.parameter_name = "tip4p_M_nonbonded_epsilon"
    nonbonded.function.sigma.parameter_name = "tip4p_M_nonbonded_sigma"

    harmonicbond = HarmonicPotential(
        equilibrium_state=r_OH,
        potential_strength=1.0,
        interaction_type="bond",
    )
    harmonicbond.equilibrium_state.parameter_name = "tip4p_OH_harmonicbond_equilibrium_state"
    harmonicbond.potential_strength.parameter_name = "tip4p_OH_harmonicbond_potential_strength"
    for interaction in universe.interactions:
        if not isinstance(interaction, Bond):
            continue
        for atm_i, atm_j in interaction.atoms:
            atm_types = sorted((atm_i.atom_type, atm_j.atom_type))
            if atm_types == ["tip4p_H", "tip4p_O"]:
                interaction.function = harmonicbond
            elif atm_types == ["tip4p_H", "tip4p_M"] or atm_types == ["tip4p_O", "tip4p_M"]:
                interaction.function = None

    harmonicangle = HarmonicPotential(
        equilibrium_state=a_HOH,
        potential_strength=1.0,
        interaction_type="angle",
    )
    harmonicangle.equilibrium_state.parameter_name = "tip4p_HOH_harmonicangle_equilibrium_state"
    harmonicangle.potential_strength.parameter_name = "tip4p_HOH_harmonicangle_potential_strength"
    for interaction in universe.interactions:
        if not isinstance(interaction, BondAngle):
            continue
        for atm_i, atm_j, atm_k in interaction.atoms:
            if sorted((atm_i.atom_type, atm_j.atom_type, atm_k.atom_type)) == [
                "tip4p_H",
                "tip4p_H",
                "tip4p_O",
            ]:
                interaction.function = harmonicangle
