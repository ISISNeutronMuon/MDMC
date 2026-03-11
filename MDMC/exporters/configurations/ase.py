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
Export structures via ASE.
"""
from typing import Any

import ase.io

from MDMC.exporters.exporter import Exporter
from MDMC.MD.ase.convert import MDMC_to_ASE
from MDMC.MD.structures import Structure


class ASEExporter(Exporter):
    """
    Use ASE to export to any format supported by ASE.
    """
    # pylint: disable=too-few-public-methods
    def write(self, obj: Structure, **settings: Any) -> None:
        """
        Write to any format supported by ASE.

        Parameters
        ----------
        obj : Structure
            The structure to export.
        **settings : str
            The format to write to. If not given, will
            be inferred from the file name.
        """

        file_format = settings.get('format')

        ase_atoms = MDMC_to_ASE(obj)
        ase.io.write(self.file, images=ase_atoms, format=file_format)
