# MDMC is a package for the optimisation of classical potentials with experimental data
# Copyright (C) 2026 MDMC Developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Reads a .cif file into an MDMC molecule, optionally ignoring symmetry data.
Adapted from https://wiki.fysik.dtu.dk/ase/_modules/ase/io/cif.html#read_cif.
"""

from typing import Any

from ase.io.cif import parse_cif

from MDMC.MD.ase.convert import ASE_to_MDMC
from MDMC.readers.configurations.ase import ASEReader
from MDMC.readers.configurations.conf_reader import ConfigurationReader


class CIFReader(ConfigurationReader):
    """
    Reads a .cif file into a list of MDMC Atoms.
    """

    extension = "cif"

    def parse(self, **settings: Any) -> None:
        """
        Parse a .cif file into a list of MDMC Atoms,
        optionally ignoring symmetry data.

        Parameters
        ----------
        settings:
            ignore_symmetry: bool, default True
                Whether to read or ignore symmetry data.
        """
        if settings.get("ignore_symmetry", True):
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
        self.finished_reading = True
