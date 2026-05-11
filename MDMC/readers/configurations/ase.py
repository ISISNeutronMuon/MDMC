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

"""MDMC wrapper for the ASE reader."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
        self._atoms: list[Atom] = []

    def parse(self, **settings: Any) -> None:
        """
        Parses any format supported by ASE's file reader; the file
        is read in as an ase.atoms.Atoms object and then converted
        to MDMC Atoms.
        """

        ASE_atoms = ase.io.read(self.file_name, **settings)
        self._atoms = ASE_to_MDMC(ASE_atoms)
