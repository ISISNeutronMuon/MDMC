"""
MDMC wrapper for the ASE reader.
"""

import ase.io

from MDMC.MD import Atom
from MDMC.MD.ase.convert import ASE_to_MDMC
from MDMC.readers.configurations.conf_reader import ConfigurationReader


class ASEReader(ConfigurationReader):
    """
    Reader that wraps around the ASE reader.

    Parameters
    ----------
    file_name : str
        File name to parse.
    """
    extension = "N/A"

    def __init__(self, file_name: str):

        super().__init__(file_name)
        self._atoms: list[Atom] = []

    def parse(self, **settings: dict) -> None:
        """
        Parse any format supported by ASE's file reader.

        The file is read in as an :any:`ase.atoms.Atoms` object and then
        converted to MDMC Atoms.

        Parameters
        ----------
        **settings : dict
            Extra options to pass to :any:`ase.io.read`.
        """

        ASE_atoms = ase.io.read(self.file_name, **settings)
        self._atoms = ASE_to_MDMC(ASE_atoms)
