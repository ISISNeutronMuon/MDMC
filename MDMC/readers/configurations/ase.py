"""MDMC wrapper for the ASE reader."""
from typing import TYPE_CHECKING, List

import ase.io

from MDMC.MD.ase.convert import ASE_to_MDMC
from MDMC.readers.configurations.conf_reader import ConfigurationReader

if TYPE_CHECKING:
    from MDMC.MD import Atom

class ASEReader(ConfigurationReader):
    """Reader that wraps around the ASE reader."""
    extension = "N/A"

    def __init__(self, file_name: str):

        super().__init__(file_name)
        self._atoms: List['Atom'] = []

    def parse(self, **settings: dict) -> None:
        """
        Parses any format supported by ASE's file reader; the file
        is read in as an ase.atoms.Atoms object and then converted
        to MDMC Atoms.
        """

        ASE_atoms = ase.io.read(self.file_name, **settings)
        self._atoms = ASE_to_MDMC(ASE_atoms)
