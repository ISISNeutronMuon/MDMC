"""Converts ASE Atoms objects into MDMC Molecules."""
from functools import reduce

import numpy as np
import ase
import ase.build as build
from ase.geometry.analysis import Analysis

from MDMC.MD import Molecule, Atom, Bond, BondAngle, DihedralAngle

def convert_ASE_to_MDMC(atoms: ase.atoms.Atoms) -> Molecule:
    """
    Convert an ase Atoms object to a Molecule.

    Parameters
    ----------
    atoms: ase.atoms.Atoms
        an ASE Atoms object.

    Returns
    -------
    Molecule
        an MDMC Molecule corresponding to the ASE Atoms object.
    """

    # first we filter the unit cell to contain just one molecule,
    # and make all positions <=0 so that we don't have to fiddle
    # with atom positions later.
    # TODO: fix this! doesn't work if atoms aren't all in the same place.
    ase_molecule = atoms
    #ase_molecule = _reduce_ase_unit_cell(atoms)
    #_make_atom_positions_valid(ase_molecule)

    # create MDMC Atom objects
    atoms_list = [Atom(atom.symbol, atom.position, charge=atom.charge) for atom in ase_molecule]

    # the ASE Analysis object contains bond information; the properties unique_bonds,
    # unique_angles and unique_dihedrals contain Bond, BondAngle and DihedralAngle
    # information respectively.
    analysis = Analysis(ase_molecule)
    interactions_list = []

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


def _reduce_ase_unit_cell(ase_atoms: 'ase.atoms.Atoms') -> ase.atoms.Atoms:
    """
    Reduces an ``ase.atoms.Atoms`` object from a unit cell of molecules to a
    single molecule

    Parameters
    ----------
    ase_atoms : ase.atoms.Atoms
        An ``ase.atoms.Atoms`` object from which a single molecule will be
        extracted

    Returns
    -------
    ASEAtoms
        An ``ASEAtoms`` object containing the atoms of a single molecule
    """

    # we simply choose all atoms connected to atom 0 in the cell
    return ase_atoms[ase.build.connected_indices(ase_atoms, 0)]


def _make_atom_positions_valid(atoms: ase.atoms.Atoms) -> None:
    """
    Sets the positions of all atoms are positive (including 0.)

    This is so that all positions are valid within ``Universe``

    Parameters
    ----------
    atoms : list
        A `list` of `Atom` which will have their positions set so that relative
        distances are preserved and the smallest positions are equal to 0.
    """

    # Offset atom positions so that they are >= 0.
    min = reduce(np.minimum, atoms.positions, [0., 0., 0.,])
    for atom in atoms:
        atom.position -= min
