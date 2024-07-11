"""
Module for reading pdb files.
"""
# pylint: disable=no-name-in-module
import itertools

import numpy as np

from MDMC.MD.interactions import Bond
from MDMC.MD.structures import Atom
from MDMC.readers.configurations.conf_reader import ConfigurationReader


class ProteinDataBankReader(ConfigurationReader):
    """
    A class for reading pdb configuration files.

    Parameters
    ----------
    file_name : str
        File to read data from.

    Examples
    --------
    To use a pdb reader to read a file called ``'water.pdb'`` and create a set of
    ``Molecule`` s from it:

    .. code-block:: python

        file = 'water.pdb'
        pdb_reader = pdb(file)
        with pdb_reader:
            pdb_reader.parse()
            water = pdb_reader.molecules[0]
    """
    extension = 'pdb'

    def __init__(self, file_name: str):
        super().__init__(file_name)
        self._bonds: list[Bond] = []

    def parse(self, **settings: dict) -> None:
        """
        Parse ``.pdb`` file into MDMC Atoms.

        Parameters
        ----------
        **settings : dict
            Extra options.

        Notes
        -----
        This follows:
        https://www.wwpdb.org/documentation/file-format v3.30 (page 180 of A4 pdf)
        Link to PDF of file format:
        https://files.wwpdb.org/pub/pdb/doc/format_descriptions/Format_v33_A4.pdf (page 180)
        """
        # pylint: disable=unused-variable
        # as molecule_id is unused but is part of the standard!
        molecule = {}
        _molecule_id = 0
        for line in self.file:
            record_name = line[0:6]
            if record_name in ("ATOM  ", "HETATM"):
                # chars 23-26 identify molecule
                atom_id = int(line[6:12].split()[-1])
                _molecule_id = int(line[22:26].split()[-1])
                element = line[76:78].split()[-1]
                current_atom_pos = [float(pos.split()[-1]) for pos in
                                    (line[30:38], line[38:46], line[46:54])]  # xyz positions
                atom_name = line[12:16].split()[-1]
                current_atom_obj = Atom(element.capitalize(), position=current_atom_pos,
                                        name=atom_name)
                self._atoms.append(current_atom_obj)
                molecule[atom_id] = current_atom_obj

            elif line[0:6] == "CONECT":
                atoms_to_connect = line[6:].split()
                # pylint: disable=no-member
                for atom1_id, atom2_id in itertools.pairwise(atoms_to_connect):
                    self.create_bond(molecule[int(atom1_id)], molecule[int(atom2_id)])

    def create_bond(self, atom1: Atom, atom2: Atom) -> None:
        """
        Create bonds in the system.

        Checks the bond lengths of the atoms in the molecule and
        creates a bond if it is below a certain threshold.

        Parameters
        ----------
        atom1 : Atom
            First atom to check.
        atom2 : Atom
            Second atom to check.

        Notes
        -----
        This is needed because PDB files are able to include H-bonds
        (which MDMC does not support) alongside other types of bonds,
        which are undistinguishable from each other in a pdb file.
        Therefore, cutting off the bond length at a reasonable
        distance prevents an extremely long bond being introduced into
        a molecule structure.

        Value of 2.1 Ang comes from:
        https://doi.org/10.1002/anie.202102967
        where 2 Ang is given as the maximum length.
        """
        cutoff = 2.1
        difference = np.subtract(atom1.position, atom2.position)
        bond_length = np.linalg.norm(difference)
        if bond_length < cutoff:
            self._bonds += [Bond((atom1, atom2))]

    @property
    def bonds(self) -> list[Bond]:
        """
        Return the bonds within the molecule.

        Bonds created by "CONECT" statements in the .pdb file and
        :any:`create_bond`.

        Returns
        -------
        list[Bond]
            Bonds in the system.
        """
        return self._bonds
