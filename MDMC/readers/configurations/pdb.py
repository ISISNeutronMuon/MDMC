"""Module for reading pdb files"""
from MDMC.readers.configurations.conf_reader import ConfigurationReader
from MDMC.MD.structures import Atom, Molecule


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

        file = 'paracetamol.pdb'
        pdb = pdb()
        pdb.open(file)
        pdb.parse()
        paracetamol = pdb.molecules[0]
    """

    def __init__(self, file_name: str):
        super().__init__(file_name)
        self._atoms = []
        self._molecules = []

    def parse(self, **settings: dict) -> None:
        super().__enter__()
        prev_id = ""
        molecules = {}
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
                    molecules[prev_id].append(current_atom)
                else:
                    if prev_id:
                        full_molecule = Molecule(atoms=molecules[prev_id])
                        self._molecules.append(full_molecule)
                    prev_id = current_id
                    molecules[prev_id] = [current_atom, ]
        super().__exit__(None, 0, None)
    @property
    def atoms(self) -> 'list[Atom]':
        return self._atoms

    @property
    def extension(self) -> str:
        return "pdb"

    @property
    def molecules(self) -> 'list[Molecule]':
        """Returns a list of ``Molecule`` objects from the data read from the file"""
        return self._molecules
