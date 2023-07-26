"""MDMC wrapper for the ASE reader."""
from typing import TYPE_CHECKING

import ase.io

from MDMC.readers.reader import Reader
from MDMC.MD.ase.convert import ASE_to_MDMC

if TYPE_CHECKING:
    from MDMC.MD import Atom

class ASEReader(Reader):
    """Reader that wraps around the ASE reader."""

    @property
    def atoms(self) -> 'list[Atom]':

        return self._atoms

    def parse(self, **settings: dict) -> None:
        """
        Parses any format supported by ASE's file reader; the file
        is read in as an ase.atoms.Atoms object and then converted
        to MDMC Atoms.
        """

        ASE_atoms = ase.io.read(self.file_name, **settings)
        self._atoms = ASE_to_MDMC(ASE_atoms)
