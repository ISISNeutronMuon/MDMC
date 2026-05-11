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

"""Module for observable reader abstract class"""

from __future__ import annotations

from typing import TYPE_CHECKING

from MDMC.common.decorators import repr_decorator
from MDMC.readers.reader import Reader

if TYPE_CHECKING:
    from MDMC.MD.structures import Atom


@repr_decorator("file", "extension", "atoms")
class ConfigurationReader(Reader):
    """
    Abstract class (as it does not implement ``Reader.parse``) that defines
    properties common to all readers for configurations

    A ``ConfigurationReader`` is created using ``ConfigurationReaderFactory``
    """

    def __init__(self, file_name: str):
        super().__init__(file_name)
        self._atoms: list[Atom] = []

    @property
    def atoms(self) -> list[Atom]:
        """
        The `Atom` objects parsed from the file.
        """

        return self._atoms
