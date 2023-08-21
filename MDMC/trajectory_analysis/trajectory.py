"""Module for ``Configuration`` and related classes"""
from typing import TYPE_CHECKING
import weakref

import numpy as np

from MDMC.common.decorators import repr_decorator

if TYPE_CHECKING:
    from MDMC.MD.simulation import Universe
    from MDMC.MD.structures import Atom, Molecule, Structure
    from typing import Any, Optional
    from builtins import function


# pylint: disable=abstract-method
# as __sub__ and scale() are not implemented


class AtomCollection:

    """Base class for ``Configurations``"""

    __slots__ = ('_universe', )

    @property
    def universe(self) -> 'Optional[Universe]':
        """
        Get or set the ``Universe`` in which the ``AtomCollection`` exists

        Returns
        -------
        Universe
            The ``Universe`` in which the ``AtomCollection`` exists, or `None`
            if it has not been set
        """

        # Call the weakref to return the universe as an object. If the use of
        # weakref causes issues with prematurely garbage collecting the
        # universe, revert this change to not use weakref.
        try:
            return self._universe()
        except TypeError:
            return None

    @universe.setter
    def universe(self, universe: 'Universe') -> None:

        # Create a weakref of the universe for _universe. If the use of weakref
        # causes issues with prematurely garbage collecting the universe,
        # revert this change to not use weakref.
        try:
            self._universe = weakref.ref(universe)
        except TypeError:
            self._universe = None

    @property
    def dimensions(self) -> 'np.ndarray':
        """
        Get the ``dimensions`` of the ``Universe`` in which the
        ``AtomCollection`` exists

        Returns
        -------
        array : numpy.ndarray
            The ``dimensions`` of the ``Universe``
        """

        return self.universe.dimensions


@repr_decorator('data')
class Configuration(AtomCollection):

    """
    A ``Configuration`` stores ``Atom`` objects and their positions and
    velocities

    Parameters
    ----------
    *structures
        Zero or more ``Structure`` objects to be added to the
        ``Configuration``
    **settings
        ``universe`` (``Universe``)
            The ``Universe`` of the ``Configuration``

    Attributes
    ----------
    element_set : set
        `set` of the elements in the ``Configuration``
    universe : Universe or None
    data : *structures
    """

    __slots__ = ('_data', 'element_set', '_structure_list')

    def __init__(self, *structures: 'Structure', **settings: dict):

        try:
            self.universe = settings['universe']
        except KeyError:
            try:
                self.universe = structures[0].universe
            except IndexError:
                self.universe = None
        self.data = structures
        self.element_set = set(self.element_list)

    def __eq__(self, other: 'Configuration') -> bool:
        if id(other) == id(self):
            return True
        if isinstance(other, self.__class__):
            for k in self.__slots__:
                if k == '_universe':
                    # As Configurations can have Universes as an attribute, and
                    # vice versa, skip comparison to prevent infinite recursion
                    continue
                v = getattr(self, k)
                try:
                    iter(v)
                    if any(v != getattr(other, k)):
                        return False
                except TypeError:
                    if v != getattr(other, k):
                        return False
            return True
        return False

    @property
    def atoms(self) -> 'list[Atom]':
        """
        Get the `list` of ``Atom`` which belong to the ``Configuration``

        Returns
        -------
        list
            A `list` of ``Atom``
        """

        return self.data['atom']

    @property
    def atom_positions(self) -> list:
        """
        Get the `list` of ``Atom.position`` which belong to the
        ``Configuration``

        Returns
        -------
        list
            A `list` of ``Atom.position``
        """
        return self.data['position']

    @property
    def atom_velocities(self) -> list:
        """
        Get the `list` of ``Atom.velocity`` which belong to the
        ``Configuration``

        Returns
        -------
        list
            A `list` of ``Atom.velocity`
        """

        return self.data['velocity']

    @property
    def element_list(self) -> list:
        """
        Get the `list` of ``Atom.element.symbol`` which belong to the ``Configuration``

        Returns
        -------
        list
            A `list` of `str` for the elements
        """

        return [atom.element.symbol for atom in self.atoms]

    @property
    def molecule_list(self) -> 'list[Molecule]':
        """
        Get the `list` of ``Molecule`` which belong to the ``Configuration``

        Returns
        -------
        list
            A `list` of ``Molecule``
        """

        return self.filter_structures(lambda x: x.structure_type == 'Molecule')

    @property
    def structure_list(self) -> 'list[Structure]':
        """
        Get the `list` of ``Structure`` which belong to the ``Configuration``

        Returns
        -------
        list
            A `list` of ``Structure``
        """

        # Call the weakref to return the structures as an object. If the
        # use of weakref causes issues with prematurely garbage collecting the
        # structures, revert this change to not use weakref.
        return [structures() for structures in self._structure_list]

    @property
    def data(self) -> np.ndarray:
        """
        Get or set the ``Atom``, ``Atom.position``, and ``Atom.velocity`` which
        belong to the ``Configuration``

        Returns
        -------
        numpy.ndarray
            A structured NumPy ``array`` with ``'atom'``, ``'position'``, and
            ``'velocity'`` fields
        """

        return np.array([(atom, atom.position, atom.velocity)
                         for atom in self._data],
                        dtype=[('atom', 'object'),
                               ('position', 'object'),
                               ('velocity', 'object')])

    @data.setter
    def data(self, structures: np.ndarray) -> None:
        self._structure_list = []
        self._data = []
        for unit in structures:
            self.add_structure(unit)

    def add_structure(self, structures: 'Structure') -> None:
        """
        Adds the ``Atom`` objects from a ``Structure`` to the data

        Parameters
        ----------
        structures : Structure
            The ``Structure`` to add
        """

        self.validate_structure(structures)
        # Create a weakref of the structures for _structure_list. If the
        # use of weakref causes issues with prematurely garbage collecting the
        # structures, revert this change to not use weakref.
        self._structure_list.append(weakref.ref(structures))
        self._data.extend(list(structures.atoms))

    def validate_structure(self, structure: 'Structure') -> None:
        """
        Validates the structure by testing that it belongs to the same
        ``Universe`` as the ``Configuration``

        Parameters
        ----------
        structure : Structure
            The ``Structure`` to validate

        Raises
        ------
        AssertionError
            If the ``Structure`` does not belong to the same ``Universe``
            as the ``Configuration``
        """

        # Test that all structural units are from the same universe
        try:
            assert structure.universe is self.universe
        except AssertionError as error:
            raise AssertionError(
                'Atoms are not all from same universe') from error

    def __add__(self, configuration: 'Configuration') -> 'Configuration':
        """
        Returns
        -------
        Configuration
            New ``Configuration`` from the sum of the ``structure_list`` of the
            two ``Configuation`` objects
        """

        structure_list = self.structure_list + configuration.structure_list

        return self.__class__(*structure_list)

    def __sub__(self, configuration: 'Configuration') -> 'Configuration':
        """
        Returns
        -------
        Configuration
            New ``Configuration`` from the difference of two ``Configuration``
            objects

        Raises
        ------
        NotImplementedError
            THIS HAS NOT BEEN IMPLEMENTED
        """

        raise NotImplementedError

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            The number of ``Atom`` objects in the ``Configuration``
        """

        return len(self.atoms)

    def __getitem__(self, item: str) -> np.ndarray:
        """
        Returns
        -------
        numpy.ndarray
            A NumPy ``array`` containing a slice from the data. The same fields
            can be accessed with ``'atom'``, ``'position'``, and ``'velocity'``.
        """

        return self.data[item]

    def filter_structures(self, predicate: 'function') -> 'list[Structure]':
        """
        Filters the `list` of ``Structures`` using the predicate

        Parameters
        ----------
        predicate : function
            A function which returns a `bool` when passed a ``Structure``

        Returns
        -------
        list
            A `list` of ``Structures`` which are `True` for the given
            predicate
        """

        return list(filter(predicate, self.structure_list))

    def filter_atoms(self, predicate: 'function') -> 'list[Atom]':
        """
        Filters the `list` of ``Atom`` using the predicate

        Parameters
        ----------
        predicate : function
            A function which returns a `bool` when passed an ``Atom``

        Returns
        -------
        list
            A `list` of ``Atom`` which are `True` for the given predicate
        """

        return list(filter(predicate, self.atoms))

    def filter_by_element(self, element: str) -> 'list[Atom]':
        """
        Filter the ``Configuration`` using an ``element``

        Parameters
        ----------
        element: str
            An elemental symbol of the same format as is used for creating
            ``Atom`` objects

        Returns
        -------
        list
            A `list` of ``Atom`` of the specified ``element``
        """

        return self.filter_atoms(lambda x: x.element == element)

    def scale(self, factor: float, vectors: str = 'positions') -> None:
        """
        Scales either ``atom_positions`` or ``atom_velocities`` by a factor

        Parameters
        ----------
        factor : float
            Factor by which the vector is scaled
        vectors : str, optional
            ``'positions'`` (default) or ``'velocities'``

        Raises
        ------
        NotImplementedError
            THIS IS NOT IMPEMENTED
        """

        raise NotImplementedError


@repr_decorator('time', 'data')
class TemporalConfiguration(Configuration):

    """
    A configuration which has a time associated with it

    Parameters
    ----------
    time : float
        The time of the ``TemporalConfiguration`` in ``fs``
    *structure_units
        Zero or more ``Structures``
    """

    __slots__ = ('time', )

    def __init__(self, time: float, *structures: 'Structure', **settings: dict) -> None:

        super().__init__(*structures, **settings)
        self.time = time

    def __add__(self, configuration: 'TemporalConfiguration') -> 'TemporalConfiguration':
        """
        Returns
        -------
        TemporalConfiguration
            New ``TemporalConfiguration`` from the sum of the
            ``TemporalConfigurations``
        """

        time = np.mean([self.time, configuration.time])

        structure_list = self.structure_list + configuration.structure_list

        return self.__class__(time, *structure_list)
