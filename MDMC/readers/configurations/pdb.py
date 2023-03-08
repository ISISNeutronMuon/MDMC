"""Module for reading pdb files"""
from itertools import pairwise

import numpy as np

from MDMC.MD.interactions import Bond
from MDMC.readers.configurations.conf_reader import ConfigurationReader
from MDMC.MD.structures import Atom


class ProteinDataBankReader(ConfigurationReader):
    """
    A class for reading pdb configuration files

    Examples
    --------
    To use a pdb reader to read a file called 'paracetamol.pdb' and create a set of
    ``Molecule``s from it (assuming ``Molecule`` has been imported from
    ``MDMC.MD``):

    .. highlight:: python
    .. code-block:: python

        file = 'water.pdb'
        pdb = pdb()
        pdb.open(file)
        pdb.parse()
        paracetamol = pdb.molecules[0]
    """

    def __init__(self, file_name: str):
        super().__init__(file_name)
        self._atoms = []
        self._molecules = []
        self._bonds = []

    def parse(self, **settings: dict) -> None:
        prev_id = ""
        self._molecule = {}
        for i in self.file:
            line = i.split()
            if line[0] == "ATOM" or line[0] == "HETATM":
                if len(line) == 9:
                    current_id = "".join(line[3:6])
                else:
                    current_id = "".join(line[3:5])

                element = line[2]
                current_pos = [float(i) for i in line[-3:]]
                current_atom = Atom(element, current_pos)
                self._atoms.append(current_atom)
                self._molecule[line[1]] = current_atom

            elif line[0] == "CONECT":
                atoms_to_connect = line[1:]
                for atom1_id, atom2_id in pairwise(atoms_to_connect):
                    self.create_bond(self._molecule[atom1_id], self._molecule[atom2_id])

    def create_bond(self, atom1: Atom, atom2: Atom) -> None:
        """
        Checks the bond lengths of the atoms in the molecule and
        creates a bond if it is below a certain threshold

        This is needed because PDB files are able to include H-bonds (which MDMC does not support)
        alongside other types of bonds, which are undistinguishable from each other in a pdb file.
        Therefore, cutting off the bond length at a reasonable distance prevents an extremely long
        bond being introduced into a molecule structure
        """
        cutoff = 2.5
        difference = np.subtract(atom1.position, atom2.position)
        bond_length = np.linalg.norm(difference)
        if bond_length < cutoff:
            self._bonds += Bond((atom1, atom2))

    @property
    def atoms(self) -> 'list[Atom]':
        return self._atoms

    def bonds(self) -> 'list[Bond]':
        return self._bonds

    @property
    def extension(self) -> str:
        return "pdb"
