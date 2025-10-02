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
