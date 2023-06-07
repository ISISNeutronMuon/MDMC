"""A reader for reading in the PDB configuration of whole packmol systems"""
import itertools

from MDMC.MD.structures import Molecule, Atom
from MDMC.readers.configurations.pdb import ProteinDataBankReader

class PackmolPDBReader(ProteinDataBankReader):
    """A class to read in packmol PDB output files"""

    def parse(self, **settings: dict) -> None:
        prev_molecule_id = ""
        molecules_dict = {}
        for line in self.file:
            # This follows https://www.wwpdb.org/documentation/file-format v3.30 (line 180 of A4 pdf)
            # Link to PDF of file format:
            # https://files.wwpdb.org/pub/pdb/doc/format_descriptions/Format_v33_A4.pdf (page 180)
            #chars 0-6 identify what the line is describing
            record_name = line[0:6]
            if record_name == "ATOM  " or record_name == "HETATM":
                #chars 23-26 identify molecule
                current_molecule_id = int(line[22:26].split()[-1])
                element = line[76:78].split()[-1]
                current_atom_pos = [float(pos.split()[-1]) for pos in
                                    (line[30:38], line[38:46], line[46:54])] #xyz positions
                atom_name = line[12:16].split()[-1]
                current_atom_obj = Atom(element, position=current_atom_pos, name=atom_name)
                self._atoms.append(current_atom_obj)

                if prev_molecule_id == current_molecule_id:
                    molecules_dict[prev_molecule_id].append(current_atom_obj)
                else:
                    if prev_molecule_id:
                        full_molecule = Molecule(atoms=molecules_dict[prev_molecule_id])
                        self._molecules.append(full_molecule)
                    prev_molecule_id = current_molecule_id
                    molecules_dict[prev_molecule_id] = [current_atom_obj,]

    @property
    def molecules(self) -> 'list[Molecule]':
        """Returns a list of ``Molecule`` objects from the data read from the file"""
        return self._molecules
