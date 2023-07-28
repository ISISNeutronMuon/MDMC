"""
Reads a .cif file into an MDMC molecule, optionally ignoring symmetry data.
Adapted from https://wiki.fysik.dtu.dk/ase/_modules/ase/io/cif.html#read_cif.
"""
from typing import TYPE_CHECKING

from ase.io.cif import parse_cif

from MDMC.MD.ase.convert import ASE_to_MDMC
from MDMC.readers.configurations.conf_reader import ConfigurationReader
from MDMC.readers.configurations.ase import ASEReader

if TYPE_CHECKING:
    from MDMC.MD import Atom

class CIFReader(ConfigurationReader):
    """
    Reads a .cif file into an MDMC molecule.
    """
    extension = 'cif'

    def parse(self, **settings: dict) -> None:
        """
        Parse a .cif file into an MDMC molecule,
        optionally ignoring symmetry data.

        Parameters
        ----------
        settings:
            ignore_symmetry: bool, default True
                Whether to read or ignore symmetry data.
        """
        if settings.get('ignore_symmetry', True):
            images = []
            for block in parse_cif(self.file):
                if not block.has_structure():
                    continue

                # this is the only place where we differ from ase.io.cif.read_cif:
                # just get the unsymmetrised structure rather than symmetrising it
                atoms = block.get_unsymmetrized_structure()
                images.append(atoms)

            atoms = list(images)[0]
            self._atoms = ASE_to_MDMC(atoms)
        else:  # else, this is just the standard ASE cif parsing
            reader = ASEReader(self.file_name)
            with reader:
                reader.parse()
                self._atoms = reader.atoms

