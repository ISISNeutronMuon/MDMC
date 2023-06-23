"""A reader for reading in the PDB configuration of whole packmol systems"""
import itertools

from MDMC.MD.structures import Molecule, Atom
from MDMC.readers.configurations.pdb import ProteinDataBankReader

class PackmolPDBReader(ProteinDataBankReader):
    """A class to read in packmol PDB output files"""

    def parse(self, **settings: dict) -> None:
        """
        Read lines from the .pdb file one at a time, following the PDB file-format. Identify the atoms, and the
        molecules to which they belong. Then add the atoms to `Molecule` objects and store in self._atoms or
        self._molecules for access by other functions.

        Parameters
        ----------
        **settings: dict, optional
            None are necessary for this reader.

        """
        prev_molecule_id = ""
        molecules_dict = {}
        full_molecule = None
        for line in self.file:
            # This follows https://www.wwpdb.org/documentation/file-format v3.30 (line 180 of A4 pdf)
            # Link to PDF of file format:
            # https://files.wwpdb.org/pub/pdb/doc/format_descriptions/Format_v33_A4.pdf (page 180)
            #chars 0-6 identify what the line is describing
            record_name = line[0:6]
            if record_name == "ATOM  " or record_name == "HETATM":
                #chars 23-26 identify molecule
                current_molecule_id = int(line[22:26].split()[-1])
                element = str(line[76:78].split()[-1])
                current_atom_pos = [float(pos.split()[-1]) for pos in
                                    (line[30:38], line[38:46], line[46:54])] #xyz positions
                atom_name = str(line[12:16].split()[-1])
                current_atom_obj = Atom(element.capitalize(), position=current_atom_pos, name=atom_name)
                self._atoms.append(current_atom_obj)

                if prev_molecule_id == current_molecule_id:
                    # We are in the same molecule so append new atom
                    molecules_dict[prev_molecule_id].append(current_atom_obj)
                else:
                    # The molecule has changed between lines - we have started to read a new molecule
                    if prev_molecule_id:
                        # A molecule has existed previously (i.e. not the first molecule)
                        full_molecule = Molecule(atoms=molecules_dict[prev_molecule_id])
                        self._molecules.append(full_molecule)
                    # Setting up for the reading a new molecule
                    # (done to allow first molecule to be read as well as between molecules)
                    prev_molecule_id = current_molecule_id
                    molecules_dict[prev_molecule_id] = [current_atom_obj,]

        #Add final molecule
        full_molecule = Molecule(atoms=molecules_dict[current_molecule_id])
        self._molecules.append(full_molecule)

    @property
    def molecules(self) -> 'list[Molecule]':
        """Returns a list of ``Molecule`` objects from the data read from the file"""
        return self._molecules
