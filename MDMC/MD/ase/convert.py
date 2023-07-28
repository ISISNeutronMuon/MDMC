"""Converts ASE Atoms objects into MDMC Molecules."""
from typing import TYPE_CHECKING, List, Union, Optional

import numpy as np
import ase
from ase.geometry.analysis import Analysis

from MDMC.MD.structures import Atom
from MDMC.MD.interactions import Bond, BondAngle, DihedralAngle

if TYPE_CHECKING:
    from MDMC.MD import Structure, Universe, BondedInteraction

def ASE_to_MDMC(atoms: ase.Atoms) -> List[Atom]:
    """
    Convert an ase Atoms object to a Molecule.

    Parameters
    ----------
    atoms: ase.Atoms
        an ASE Atoms object.

    Returns
    -------
    Molecule
        an MDMC Molecule corresponding to the ASE Atoms object.
    """

    # create MDMC Atom objects
    atoms_list = [Atom(atom.symbol, atom.position, charge=atom.charge) for atom in atoms]

    # the ASE Analysis object contains bond information; the properties unique_bonds,
    # unique_angles and unique_dihedrals contain Bond, BondAngle and DihedralAngle
    # information respectively.
    analysis = Analysis(ase_molecule)
    interactions_list: List['BondedInteraction'] = []

    # ase bond lists have the following structure:
    # index X of the list contains all bonds that start at atom number X.
    # e.g. if index 0 of unique_angles contains (1, 3) that means
    # there is a bond angle of atoms 0—1—3
    # (the same order as MDMC)
    # one bond list is generated per neighbour list; we should only
    # have one neighbour list here.
    for index, bonds in enumerate(analysis.unique_bonds[0]):
        interactions_list.extend([Bond(atoms_list[index],
                                       atoms_list[bonded_atom])
                                       for bonded_atom in bonds])

    for index, bonds in enumerate(analysis.unique_angles[0]):
        interactions_list.extend([BondAngle(atoms_list[index],
                                            atoms_list[bonded_atoms[0]],
                                            atoms_list[bonded_atoms[1]])
                                            for bonded_atoms in bonds])

    for index, bonds in enumerate(analysis.unique_dihedrals[0]):
        interactions_list.extend([DihedralAngle(atoms_list[index],
                                                atoms_list[bonded_atoms[0]],
                                                atoms_list[bonded_atoms[1]],
                                                atoms_list[bonded_atoms[2]])
                                                for bonded_atoms in bonds])

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

    return ase.atom.Atom(position=atom.position,
                         mass=atom.mass,
                         symbol=atom.element,
                         charge=atom.charge)


def MDMC_to_ASE(structure: Union['Structure', 'Universe'],
                cell: Optional[np.ndarray] = None) -> ase.Atoms:
    """
    Convert an MDMC Structure into an ase.Atoms object.
    Note that ASE infers bonds from the atoms' covalent radius.

    Parameters:
    -----------
    structure: Structure
        the MDMC Structure to convert.
    cell: np.array, optional, default None
        provides cell dimensions for the ASE Atoms object.
        If None, the default cell size (0,0,0) is used.

    Returns:
    --------
    ase.Atoms
        an ASE Atoms object corresponding to the same structure.
    """
    if cell is None:
        cell = np.array([0., 0., 0.,])

    return ase.Atoms([_convert_to_ase_atom(atom) for atom in structure.atoms], cell=cell)
