"""A reader for reading in the PDB configuration of whole packmol systems"""
from MDMC.MD import Atom, Molecule
from MDMC.readers.reader import Reader

class PackmolPDBReader(Reader):

    def __init__(self, file_name: str):

        super().__init__(file_name)
        self._atoms = []
        self._molecules = []

    def parse(self, **settings: dict) -> None:
        prev_id = ""
        molecules_dict = {}
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

                if prev_id == current_id:
                    molecules_dict[prev_id].append(current_atom)
                else:
                    if prev_id:
                        full_molecule = Molecule(atoms=molecules_dict[prev_id])
                        self._molecules.append(full_molecule)
                    prev_id = current_id
                    molecules_dict[prev_id] = [current_atom, ]


    @property
    def atoms(self) -> 'list[Atom]':
        return self._atoms

    @property
    def molecules(self) -> 'list[Molecule]':
        """Returns a list of ``Molecule`` objects from the data read from the file"""
        return self._molecules