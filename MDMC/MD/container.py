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
"""Module for the AtomContainer class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from MDMC.MD.structures import Atom


class AtomContainer(ABC):
    """
    A collection of ``Atom`` objects

    The ``AtomContainer`` can be indexed and returns an ``Atom``. However
    indexing cannot be used for setting or deleting an `Atom`.

    Attributes
    ----------
    atoms : list
        A list of the ``Atom`` objects that belong to the ``AtomContainer``
    """

    @property
    @abstractmethod
    def atoms(self) -> list[Atom]:
        """
        Returns
        -------
        list
            A list of the ``Atom`` objects belonging to the ``AtomContainer``
        """

        raise NotImplementedError

    @overload
    def __getitem__(self, index: int) -> Atom: ...
    @overload
    def __getitem__(self, index: slice) -> list[Atom]: ...
    def __getitem__(self, index):
        """
        Returns
        -------
        Atom, list
            The atom (or atoms) for the specified index (or slice)
        """

        return self.atoms[index]
