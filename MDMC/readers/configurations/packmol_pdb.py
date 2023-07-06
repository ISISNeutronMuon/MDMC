"""A reader for reading in the PDB configuration of whole packmol systems"""
from typing import List, Dict, TYPE_CHECKING

from MDMC.MD.structures import Molecule, Atom
from MDMC.readers.configurations.pdb import ProteinDataBankReader

if TYPE_CHECKING:
    from MDMC.MD.structures import Structure

class PackmolPDBReader(ProteinDataBankReader):
    """A class to read in packmol PDB output files"""

    def parse(self, **settings: dict) -> None:
        """
        Parses a .pdb file into a list of `Structure`s (atoms and molecules)
        which can then be accessed as the `structures` property of the reader.

        Parameters
        ----------
        **settings: dict, optional
            None are necessary for this reader.

        """
        molecules_dict: Dict[str, List[Atom]] = {}
        for line in self.file:
            #chars 0-6 identify what the line is describing
            record_name = line[0:6]
            if record_name in ("ATOM  ", "HETATM"):
                record = self._parse_atom_record(line)
                atom_molecule_id = record['molecule_id']
                current_atom_obj = Atom(record['element_symbol'].capitalize(),
                                        position=record['atom_position'],
                                        name=record['name'])

                if atom_molecule_id in molecules_dict:
                    molecules_dict[atom_molecule_id].append(current_atom_obj)
                else:
                    molecules_dict[atom_molecule_id] = [current_atom_obj]

        # if multiple atoms share an id, this means they are part of the same
        # molecule; create a Molecule object for them
        for atom_list in molecules_dict.values():
            if len(atom_list) == 1:
                self._structures.append(atom_list[0])
            else:
                self._structures.append(Molecule(atoms=atom_list))


    def _parse_atom_record(self, line):
        """
        A function to get the necessary information out of an `ATOM  ` or `HETATM` record
        This follows https://www.wwpdb.org/documentation/file-format v3.30 (page 180 of A4 pdf)
        Link to PDF of file format:
        https://files.wwpdb.org/pub/pdb/doc/format_descriptions/Format_v33_A4.pdf (page 180)
        """
        record = {
            'name': line[12:16].split()[-1],
            'molecule_id': int(line[22:26]),
            'atom_position': tuple(float(pos) for pos in [line[30:38],
                                                          line[38:46],
                                                          line[46:54]]),
            'element_symbol': line[76:78].split()[-1]
        }
        return record

    @property
    def structures(self) -> List['Structure']:
        """Returns a list of ``Molecule`` objects from the data read from the file"""
        return self._structures
