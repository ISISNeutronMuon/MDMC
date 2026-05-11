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

"""Converts ASE Atoms objects into MDMC Molecules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ase
import numpy as np
from ase.geometry.analysis import Analysis

from MDMC.MD.interactions import Bond, BondAngle, DihedralAngle
from MDMC.MD.structures import Atom, Molecule

if TYPE_CHECKING:
    from MDMC.MD import BondedInteraction, Structure, Universe


def ASE_to_MDMC(atoms: ase.Atoms) -> list[Atom]:
    """
    Convert an ASE Atoms object to a list of Atoms.

    Parameters
    ----------
    atoms: ase.Atoms
        an ASE Atoms object.

    Returns
    -------
    List[Atom]
        a list of Atoms with bonds corresponding to the ASE Atoms object.
    """

    # create MDMC Atom objects
    atoms_list = [Atom(atom.symbol, atom.position, charge=atom.charge) for atom in atoms]

    # the ASE Analysis object contains bond information; the properties unique_bonds,
    # unique_angles and unique_dihedrals contain Bond, BondAngle and DihedralAngle
    # information respectively.
    analysis = Analysis(atoms)
    interactions_list: list[BondedInteraction] = []

    # ase bond lists have the following structure:
    # index X of the list contains all bonds that start at atom number X.
    # e.g. if index 0 of unique_angles contains (1, 3) that means
    # there is a bond angle of atoms 0—1—3
    # (the same order as MDMC)
    # one bond list is generated per neighbour list; we should only
    # have one neighbour list here.
    for index, bonds in enumerate(analysis.unique_bonds[0]):
        interactions_list.extend(
            [Bond(atoms_list[index], atoms_list[bonded_atom]) for bonded_atom in bonds],
        )

    for index, bonds in enumerate(analysis.unique_angles[0]):
        interactions_list.extend(
            [
                BondAngle(
                    atoms_list[index],
                    atoms_list[bonded_atoms[0]],
                    atoms_list[bonded_atoms[1]],
                )
                for bonded_atoms in bonds
            ],
        )

    for index, bonds in enumerate(analysis.unique_dihedrals[0]):
        interactions_list.extend(
            [
                DihedralAngle(
                    atoms_list[index],
                    atoms_list[bonded_atoms[0]],
                    atoms_list[bonded_atoms[1]],
                    atoms_list[bonded_atoms[2]],
                )
                for bonded_atoms in bonds
            ],
        )

    return atoms_list


def _convert_to_ase_atom(atom: Atom) -> ase.Atom:
    """
    Converts an MDMC ``Atom`` to an ``ase.Atom``

    Parameters
    ----------
    atom : Atom
        An MDMC ``Atom`` object to be converted to an ``ase.Atom`` object

    Returns
    -------
    ase.atom.Atom
        An ``ASE.Atom`` object which is equivalent to ``atom``
    """

    return ase.atom.Atom(
        position=atom.position,
        mass=atom.mass,
        symbol=atom.element.symbol,
        charge=atom.charge,
    )


def MDMC_to_ASE(
    structure: Structure | Universe | list[Atom],
    cell: np.ndarray | None = None,
) -> ase.Atoms:
    """
    Convert an MDMC Structure into an ase.Atoms object.
    Note that ASE infers bonds from the atoms' covalent radius.

    Parameters:
    -----------
    structure: Structure, Universe or List[Atom]
        the MDMC object to convert.
    cell: np.ndarray, optional, default None
        provides cell dimensions for the ASE Atoms object.
        If None, the default cell size (0,0,0) is used.

    Returns:
    --------
    ase.Atoms
        an ASE Atoms object corresponding to the same structure.
    """
    if isinstance(structure, list):  # if list of atoms, compile into molecule
        structure = Molecule(atoms=structure)

    if cell is None:
        cell = np.array([0.0, 0.0, 0.0])

    return ase.Atoms([_convert_to_ase_atom(atom) for atom in structure.atoms], cell=cell)
