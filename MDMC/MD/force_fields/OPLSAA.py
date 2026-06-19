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

import itertools as it

import pandas as pd

from MDMC.MD.force_fields.ff import FileForceField
from MDMC.MD.interaction_functions import HarmonicPotential, NonBonded, Periodic
from MDMC.MD.interactions import Bond, BondAngle, DihedralAngle, NonBondedForce
from MDMC.MD.structures import Atom


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
        atom_types.add(atom.atom_type)  # mdmc atom type

    type_to_name = {atom.atom_type: atom.name for atom in universe.atoms}
    atoms_df = opls_aa_file.atoms
    disp_df = opls_aa_file.dispersions
    for atom_type in atom_types:
        name = type_to_name[atom_type]  # name = OPLS atom type
        charge = atoms_df[atoms_df["atom_type"] == int(name)].iloc[0]["charge"]
        disp_row = disp_df[disp_df["atom_type"] == int(name)].iloc[0]
        nonbonded = NonBondedForce(
            universe,
            atom_type,
            cutoff=cutoff,
            ewald=ewald,
            function=NonBonded(charge=charge, epsilon=disp_row["epsilon"], sigma=disp_row["sigma"]),
        )
        nonbonded.function.charge.parameter_name = f"OPLS-{name}-nonbonded_charge"
        nonbonded.function.epsilon.parameter_name = f"OPLS-{name}-nonbonded_epsilon"
        nonbonded.function.sigma.parameter_name = f"OPLS-{name}-nonbonded_sigma"
        universe.set_atom_charge(atom_type, charge)

    bonds_df = opls_aa_file.bonds
    angles_df = opls_aa_file.bond_angles
    impropers_df = opls_aa_file.impropers
    propers_df = opls_aa_file.propers

    for interaction in universe.interactions:
        if isinstance(interaction, Bond):
            bond_row, opls_str = find_row(atoms_df, bonds_df, *interaction.atoms[0])

            harmonicbond = HarmonicPotential(
                equilibrium_state=bond_row["equilibrium_state"],
                potential_strength=bond_row["potential_strength"],
                interaction_type="bond",
            )
            harmonicbond.equilibrium_state.parameter_name = (
                f"{opls_str}-harmonicbond_equilibrium_state"
            )
            harmonicbond.potential_strength.parameter_name = (
                f"{opls_str}-harmonicbond_potential_strength"
            )
            interaction.function = harmonicbond

        elif isinstance(interaction, BondAngle):
            angle_row, opls_str = find_row(atoms_df, angles_df, *interaction.atoms[0])

            harmonicangle = HarmonicPotential(
                equilibrium_state=angle_row["equilibrium_state"],
                potential_strength=angle_row["potential_strength"],
                interaction_type="angle",
            )
            harmonicangle.equilibrium_state.parameter_name = (
                f"{opls_str}-harmonicangle_equilibrium_state"
            )
            harmonicangle.potential_strength.parameter_name = (
                f"{opls_str}-harmonicangle_potential_strength"
            )
            interaction.function = harmonicangle

        elif isinstance(interaction, DihedralAngle):
            atm_i, atm_j, atm_k, atm_l = interaction.atoms[0]

            if interaction.improper:
                dihedral_row, opls_str = find_row(
                    atoms_df, impropers_df, atm_i, atm_j, atm_k, atm_l, check_reverse=False
                )

                dihedral = Periodic(
                    # the force constant oplsaa.dat is defined for a fourier
                    # type potential with a factor of 1/2 in front. MDMC
                    # Periodic function does not have a factor of 1/2
                    # according to the docstring
                    dihedral_row["K1"] / 2,
                    int(dihedral_row["n1"]),
                    dihedral_row["d1"],
                )
                dihedral.K1.parameter_name = f"{opls_str}-dihedral-improper-K1"
                dihedral.n1.parameter_name = f"{opls_str}-dihedral-improper-n1"
                dihedral.d1.parameter_name = f"{opls_str}-dihedral-improper-d1"
            else:
                dihedral_row, opls_str = find_row(atoms_df, propers_df, atm_i, atm_j, atm_k, atm_l)

                dihedral = Periodic(
                    # need to divide by 2 see above
                    dihedral_row["K1"] / 2,
                    int(dihedral_row["n1"]),
                    dihedral_row["d1"],
                    dihedral_row["K2"] / 2,
                    int(dihedral_row["n2"]),
                    # in opls the second terms have a minus in front of
                    # the cosine. in openmm the torsion force has a positive
                    # sign in front, the parameter files should take account
                    # of this
                    dihedral_row["d2"],
                    dihedral_row["K3"] / 2,
                    int(dihedral_row["n3"]),
                    dihedral_row["d3"],
                )
                dihedral.K1.parameter_name = f"{opls_str}-dihedral-proper-K1"
                dihedral.K2.parameter_name = f"{opls_str}-dihedral-proper-K2"
                dihedral.K3.parameter_name = f"{opls_str}-dihedral-proper-K3"
                dihedral.n1.parameter_name = f"{opls_str}-dihedral-proper-n1"
                dihedral.n2.parameter_name = f"{opls_str}-dihedral-proper-n2"
                dihedral.n3.parameter_name = f"{opls_str}-dihedral-proper-n3"
                dihedral.d1.parameter_name = f"{opls_str}-dihedral-proper-d1"
                dihedral.d2.parameter_name = f"{opls_str}-dihedral-proper-d2"
                dihedral.d3.parameter_name = f"{opls_str}-dihedral-proper-d3"

            interaction.function = dihedral


def find_row(atoms_df: pd.DataFrame, params_df: pd.DataFrame, *args: Atom, check_reverse=True):
    """From the OPLS parameter data frame get the data frame row for the
    inputted atoms. Uses wildcards atom groups if required. A parameter
    set which uses a fewer number of wildcard atom groups are given
    higher priority than those with more.

    Parameters
    ----------
    atoms_df : pd.DataFrame
        Data frame of OPLS atom types.
    params_df : pd.DataFrame
        Data frame of OPLS force field parameters.
    *args : Atom
        The atoms to find OPLS parameters for.
    check_reverse : bool
        Check the reversed order atoms.

    Raises
    ------
    ValueError
        If a parameter set is unable to be found or if multiple parameter
        sets are found for the same number of wildcard atom groups used.

    """
    atm_types = [int(atm.name) for atm in args]
    switches = sorted(it.product([0, 1], repeat=len(args)), key=sum, reverse=True)
    switches = [list(group) for _, group in it.groupby(switches, key=sum)]

    # Switch to change the atoms type to a wildcard type. We first check
    # with the most specific first, if that is not found we change
    # one of them to a wildcard atom type and check again.
    for switch_grp in switches:
        # build up a list of atm groups ids that we need to check for
        # all combinations with the same number of wildcard atom types
        atm_grps = []
        for switch in switch_grp:
            atm_grps_sublist = []
            for i, atm_type in enumerate(atm_types):
                s_i = atoms_df["atom_type"] == atm_type
                grp_i = int(atoms_df[s_i].iloc[0]["atom_group"] * switch[i])
                atm_grps_sublist.append(grp_i)
            atm_grps.append(atm_grps_sublist)
            if check_reverse:
                atm_grps.append(list(reversed(atm_grps_sublist)))
        atm_grps = [list(x) for x in {tuple(y) for y in atm_grps}]

        # get all matches in the dataframe
        rows_result = []
        grp_result = []
        for atm_grps_sublist in atm_grps:
            selections = []
            for i, grp_i in enumerate(atm_grps_sublist):
                selections.append(params_df[f"atom_group{i + 1}"] == grp_i)
            try:
                new_df = params_df
                for s in selections:
                    new_df = new_df[s]
                rows_result.append(new_df.iloc[0])
                grp_result.append([str(grp) for grp in atm_grps_sublist])
            except IndexError:
                pass

        # return result if there is only one match
        if len(rows_result) == 1:
            return rows_result[0], "OPLS-" + "-".join(grp_result[0])
        elif len(rows_result) > 1:
            raise ValueError(
                f"Multiple parameter sets found for atom types {atm_types} "
                f"with groups {grp_result}, when {sum(switch_grp[0])} wildcard atom "
                f"groups are used."
            )

    raise ValueError(f"Unable to find parameters set for atom types: {atm_types}")
