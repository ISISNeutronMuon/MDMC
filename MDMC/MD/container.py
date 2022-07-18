"""Module for the AtomContainer class
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

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
    def atoms(self) -> 'list[Atom]':
        """
        Returns
        -------
        list
            A list of the ``Atom`` objects belonging to the ``AtomContainer``
        """

        raise NotImplementedError

    def __getitem__(self, index: Union[int, slice]) -> Union['Atom', 'list[Atom]']:
        """
        Returns
        -------
        Atom, list
            The atom (or atoms) for the specified index (or slice)
        """

        return self.atoms[index]
