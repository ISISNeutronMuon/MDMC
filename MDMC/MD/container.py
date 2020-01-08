"""Module for the AtomContainer class
"""

from abc import ABC, abstractmethod


class AtomContainer(ABC):

    """
    A collection of ``Atom`` objects

    The ``AtomContainer`` has a length equal to the number of ``Atom`` objects,
    and can also be indexed. However indexing cannot be used for setting or
    deleting an `Atom`.

    Attributes
    ----------
    atom_list : list
        A list of the ``Atom`` objects that belong to the ``AtomContainer``
    """

    @property
    @abstractmethod
    def atom_list(self):

        """
        Returns
        -------
        list
            A list of the ``Atom`` objects belonging to the ``AtomContainer``
        """

        raise NotImplementedError

    def __len__(self):

        return len(self.atom_list)

    def __getitem__(self, index):

        """
        Returns
        -------
        Atom, list
            The atom (or atoms) for the specified index (or slice)
        """

        return self.atom_list[index]
