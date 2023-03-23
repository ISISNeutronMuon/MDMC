"""A reader for reading in the PDB configuration of whole packmol systems"""
import itertools

from MDMC.MD.structures import Molecule, Atom
from MDMC.readers.configurations.pdb import ProteinDataBankReader

class PackmolPDBReader(ProteinDataBankReader):
    """A class to read in packmol """

    def parse(self, **settings: dict) -> None:
        prev_molecule_id = ""
        atom_id_dict = {}
        molecules_dict = {}
        for i in self.file:
            line = i.split()
            if line[0] == "ATOM" or line[0] == "HETATM":
                if len(line) == 9:
                    current_molecule_id = "".join(line[3:6])
                else:
                    current_molecule_id = "".join(line[3:5])

                element = line[2]
                current_atom_pos = [float(i) for i in line[-3:]]
                current_atom_obj = Atom(element, current_atom_pos)
                atom_id_dict[line[1]] = current_atom_obj

                if prev_molecule_id == current_molecule_id:
                    molecules_dict[prev_molecule_id].append(current_atom_obj)
                else:
                    if prev_molecule_id:
                        full_molecule = Molecule(atoms=molecules_dict[prev_molecule_id])
                        self._molecules.append(full_molecule)
                    prev_molecule_id = current_molecule_id
                    molecules_dict[prev_molecule_id] = [current_atom_obj, ]

            elif line[0] == "CONECT":
                atoms_to_connect = line[1:]
                for atom1_id, atom2_id in itertools.pairwise(atoms_to_connect):
                    self.create_bond(atom_id_dict[atom1_id], atom_id_dict[atom2_id])

        self._atoms = atom_id_dict.values()

    @property
    def molecules(self) -> 'list[Molecule]':
        """Returns a list of ``Molecule`` objects from the data read from the file"""
        return self._molecules
