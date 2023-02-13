"""Module for reading pdb files"""
from typing import TYPE_CHECKING

from MDMC.common.decorators import set_docstring
from MDMC.readers.configurations.conf_reader import ConfigurationReader

if TYPE_CHECKING:
    from MDMC.MD.structures import Atom


class ProteinDataBankReader(ConfigurationReader):

    """
    A class for reading pdb configuration files

    Examples
    --------
    To use a pdb reader to read a file called 'paracetamol.pdb' and create a
    ``Molecule`` from it (assuming ``Molecule`` has been imported from
    ``MDMC.MD``):

    .. highlight:: python
    .. code-block:: python

        file = 'paracetamol.pdb'
        pdb = pdb()
        pdb.open(file)
        # See parse docstring for description of ``names`` parameter
        # Lines are oxygen, nitrogen, carbon and hydrogen atoms
        pdb.parse(names=['109', '177', # Oxygens
                         '207', # Nitrogen
                         '208', '108', '90', '178', '90', '90', '90', '185',
                         '85', '85', '85', '91', '91', '91', '91', '183', '110']
                 ) # Hydrogens)
        paracetamol = Molecule(atoms=pdb.atoms)
    """

    extension = 'pdb'

    def __init__(self, file_name: str):
        super().__init__(file_name)
        with open(file_name, "r") as f:
            file_contents = f.readlines()

        molecules = {}
        prev_id = ""
        for i in file_contents:
            line = i.split()
            if line[0] == "ATOM" or line[0] == "HETATM":
                current_id = "".join(line[3:6])
                if prev_id == current_id:
                    molecules[prev_id].append(line[-4:-1])
                else:
                    prev_id = current_id
                    molecules[prev_id] = [line[-4:-1], ]
        self._atoms = None


    def parse(self, **settings: dict) -> None:



    @property
    def atoms(self) -> 'list[Atom]':

        return self._atoms
