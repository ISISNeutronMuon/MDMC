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

"""A module for defining the OPLSAA force field. This was generated from the
corresponding TINKER file."""


from MDMC.MD.force_fields.ff import FileForceField
from MDMC.MD.interaction_functions import HarmonicPotential, NonBonded, Periodic
from MDMC.MD.interactions import Bond, BondAngle, DihedralAngle, NonBondedForce


class OPLSAA(FileForceField):
    """
    OPLSAA force field, with defined atoms and interactions
    """

    file_name = "oplsaa.dat"


def add_opls_force_field(universe, cutoff: float, ewald: float):
    """Adds the OPLS force field to the universe.

    Parameters
    ----------
    universe : Universe
        The MDMC universe object.
    cutoff : float
        The cutoff distance (angstrom) used for nonbonded interactions.
    ewald : float
        The error tolerance for Ewald summation.
    """
    opls_aa_file = OPLSAA()

    atom_types = set()
    for atom in universe.atoms:
        atom_types.add(atom.atom_type)

    atoms_df = opls_aa_file.atoms
    disp_df = opls_aa_file.dispersions
    for atom_type in atom_types:
        charge = atoms_df[atoms_df["atom_type"] == int(atom_type)].iloc[0]["charge"]
        disp_row = disp_df[disp_df["atom_type"] == int(atom_type)].iloc[0]
        nonbonded = NonBondedForce(
            universe,
            atom_type,
            cutoff=cutoff,
            ewald=ewald,
            function=NonBonded(charge=charge, epsilon=disp_row["epsilon"], sigma=disp_row["sigma"]),
        )
        nonbonded.function.charge.parameter_name = f"OPLS-{atom_type}-nonbonded_charge"
        nonbonded.function.epsilon.parameter_name = f"OPLS-{atom_type}-nonbonded_epsilon"
        nonbonded.function.sigma.parameter_name = f"OPLS-{atom_type}-nonbonded_sigma"

    bonds_df = opls_aa_file.bonds
    angles_df = opls_aa_file.bond_angles
    impropers_df = opls_aa_file.impropers
    propers_df = opls_aa_file.propers

    for interaction in universe.interactions:
        if isinstance(interaction, Bond):
            atm_i, atm_j = interaction.atoms[0]

            s_i = atoms_df["atom_type"] == int(atm_i.atom_type)
            s_j = atoms_df["atom_type"] == int(atm_j.atom_type)
            grp_i = atoms_df[s_i].iloc[0]["atom_group"]
            grp_j = atoms_df[s_j].iloc[0]["atom_group"]
            grp_i, grp_j = sorted([grp_i, grp_j])

            s_i = bonds_df["atom_group1"] == grp_i
            s_j = bonds_df["atom_group2"] == grp_j
            bond_row = bonds_df[s_i][s_j].iloc[0]

            harmonicbond = HarmonicPotential(
                equilibrium_state=bond_row["equilibrium_state"],
                potential_strength=bond_row["potential_strength"],
                interaction_type="bond",
            )
            harmonicbond.equilibrium_state.parameter_name = (
                f"OPLS-{grp_i}-{grp_j}-harmonicbond_equilibrium_state"
            )
            harmonicbond.potential_strength.parameter_name = (
                f"OPLS-{grp_i}-{grp_j}-harmonicbond_potential_strength"
            )
            interaction.function = harmonicbond

        elif isinstance(interaction, BondAngle):
            atm_i, atm_j, atm_k = interaction.atoms[0]

            s_i = atoms_df["atom_type"] == int(atm_i.atom_type)
            s_j = atoms_df["atom_type"] == int(atm_j.atom_type)
            s_k = atoms_df["atom_type"] == int(atm_k.atom_type)
            grp_i = atoms_df[s_i].iloc[0]["atom_group"]
            grp_j = atoms_df[s_j].iloc[0]["atom_group"]
            grp_k = atoms_df[s_k].iloc[0]["atom_group"]
            if grp_i > grp_k:
                grp_i, grp_j, grp_k = reversed([grp_i, grp_j, grp_k])

            s_i = angles_df["atom_group1"] == grp_i
            s_j = angles_df["atom_group2"] == grp_j
            s_k = angles_df["atom_group3"] == grp_k
            angle_row = angles_df[s_i][s_j][s_k].iloc[0]

            harmonicangle = HarmonicPotential(
                equilibrium_state=angle_row["equilibrium_state"],
                potential_strength=angle_row["potential_strength"],
                interaction_type="angle",
            )
            harmonicangle.equilibrium_state.parameter_name = (
                f"OPLS-{grp_i}-{grp_j}-{grp_k}-harmonicangle_equilibrium_state"
            )
            harmonicangle.potential_strength.parameter_name = (
                f"OPLS-{grp_i}-{grp_j}-{grp_k}-harmonicangle_potential_strength"
            )
            interaction.function = harmonicangle

        elif isinstance(interaction, DihedralAngle):
            atm_i, atm_j, atm_k, atm_l = interaction.atoms[0]

            s_i = atoms_df["atom_type"] == int(atm_i.atom_type)
            s_j = atoms_df["atom_type"] == int(atm_j.atom_type)
            s_k = atoms_df["atom_type"] == int(atm_k.atom_type)
            s_l = atoms_df["atom_type"] == int(atm_l.atom_type)
            grp_i = atoms_df[s_i].iloc[0]["atom_group"]
            grp_j = atoms_df[s_j].iloc[0]["atom_group"]
            grp_k = atoms_df[s_k].iloc[0]["atom_group"]
            grp_l = atoms_df[s_l].iloc[0]["atom_group"]

            if (grp_i == grp_l and grp_j > grp_k) or grp_i > grp_l:
                grp_i, grp_j, grp_k, grp_l = reversed([grp_i, grp_j, grp_k, grp_l])

            if interaction.improper:
                dihedral_df = impropers_df
                type_str = "improper"
            else:
                dihedral_df = propers_df
                type_str = "proper"

            s_i = dihedral_df["atom_group1"] == grp_i
            s_j = dihedral_df["atom_group2"] == grp_j
            s_k = dihedral_df["atom_group3"] == grp_k
            s_l = dihedral_df["atom_group4"] == grp_l
            dihedral_row = dihedral_df[s_i][s_j][s_k][s_l].iloc[0]

            dihedral = Periodic(
                dihedral_row["K1"],
                int(dihedral_row["n1"]),
                dihedral_row["d1"],
                dihedral_row["K2"],
                int(dihedral_row["n2"]),
                dihedral_row["d2"],
                dihedral_row["K3"],
                int(dihedral_row["n3"]),
                dihedral_row["d3"],
            )
            opls_str = f"OPLS-{grp_i}-{grp_j}-{grp_k}-{grp_l}"
            dihedral.K1.parameter_name = f"{opls_str}-dihedral-{type_str}-K1"
            dihedral.K2.parameter_name = f"{opls_str}-dihedral-{type_str}-K2"
            dihedral.K3.parameter_name = f"{opls_str}-dihedral-{type_str}-K3"
            dihedral.n1.parameter_name = f"{opls_str}-dihedral-{type_str}-n1"
            dihedral.n2.parameter_name = f"{opls_str}-dihedral-{type_str}-n2"
            dihedral.n3.parameter_name = f"{opls_str}-dihedral-{type_str}-n3"
            dihedral.d1.parameter_name = f"{opls_str}-dihedral-{type_str}-d1"
            dihedral.d2.parameter_name = f"{opls_str}-dihedral-{type_str}-d2"
            dihedral.d3.parameter_name = f"{opls_str}-dihedral-{type_str}-d3"
            interaction.function = dihedral
