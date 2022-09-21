"""Module for ``Configuration`` and ``Trajectory`` classes, and related classes"""
from typing import TYPE_CHECKING, Union
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

    """Base class for shared attributes for ``Configurations`` and ``Trajectories``"""

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
        Get the `list` of ``Atom.element`` which belong to the ``Configuration``

        Returns
        -------
        list
            A `list` of `str` for the elements
        """

        return [atom.element for atom in self.atoms]

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


@repr_decorator('times', 'data')
class Trajectory(AtomCollection):

    """
    A ``Trajectory`` is a collection of ``TemporalConfigurations``

    Parameters
    ----------
    *configurations
        One or more ``TemporalConfigurations``

    Attributes
    ----------
    configurations : list
        A `list` of ``TemporalConfigurations``
    """

    __slots__ = ('_data', )

    def __init__(self, *configurations: 'TemporalConfiguration') -> None:

        # Check that each configuration has the same universe
        try:
            universe = configurations[0].universe
        except IndexError as error:
            raise ValueError('At least one ``TemporalConfiguration`` must be'
                             'provided') from error

        for configuration in configurations[1:]:
            if configuration.universe != universe:
                raise ValueError('The universes of all TemporalConfigurations'
                                 f'are not equivalent:\n{universe}\n{configuration.universe}')

        self.data = configurations

    def __getstate__(self) -> dict:
        """
        Gets the state of the Trajectory so that it can be pickled. The ``_structure_list``
        attribute of a ``TemporalConfiguration`` and ``self._universe`` are both weak references,
        which cannot be pickled. Therefore we must create a custom dictionary that contains the
        original objects the weak references were pointing to.

        Returns
        -------
        dict
            Dictionary containing all the necessary objects to define a ``Trajectory`` without
            including any weak references.
        """

        structures = [config.structure_list for config in self.configurations]

        return {'universe': self.universe,
                'times': self.times,
                'structures': structures}

    def __setstate__(self, d: dict) -> None:
        """
        Sets the state of the Trajectory when it is being  unpickled. The ``_structure_list``
        attribute of a ``TemporalConfiguration`` and ``self._universe`` are both weak references,
        which cannot be pickled. We load the custom dictionary ``d`` that contains objects and then
        use it to initialise the ``TemporalConfiguration`` and ``Trajectory``. In doing so, their
        attributes are populated with weak references to the objects in ``d``.

        Parameters
        ----------
        d, dict
            Dictionary containing all the necessary objects to define a ``Trajectory`` without
            including any weak references.
        """

        configs = [TemporalConfiguration(time, *structure, universe=d['universe'])
                   for time, structure in zip(d['times'], d['structures'])]

        self.__init__(*configs)

    @property
    def data(self) -> np.ndarray:
        """
        Get or set the data of the ``Trajectory``

        Returns
        -------
        numpy.ndarray
            An ordered ``array`` of ``frames``, ``times`` (in ``fs``) and
            ``TemporalConfigurations``
        """

        return self._data

    @data.setter
    def data(self, configurations: np.ndarray) -> None:
        self._data = np.array(
            [(i, config.time, config) for
             i, config in enumerate(configurations, 1)],
            dtype=[('frame', 'int64'),
                   ('time', 'float64'),
                   ('configuration', 'object')])

        for datum in self._data:
            if datum['frame'] == 1:
                config0 = datum['configuration']
            else:
                self.validate_config(datum['configuration'], validator=config0)

    @staticmethod
    def validate_config(config: 'Configuration', validator: 'Configuration') -> None:
        """
        Validates that a ``Configuration`` has the same number of ``Atom``
        objects as the validator

        Parameters
        ----------
        config : Configuration
            The ``Configuration`` to test
        validator : Configuration
            The ``Configuration`` against which to compare ``config``

        Raises
        ------
        AssertionError
            If the number of ``Atom`` objects in the ``Configurations`` do not match
        """

        try:
            assert len(config.atoms) == len(validator.atoms)
        except AssertionError as error:
            raise AssertionError('Configurations do not contain the same number'
                                 ' of atoms') from error

    def __getitem__(self, item: Union[str, int]) -> 'Any':
        """Indexing and slicing is relative to ``frames``"""

        if isinstance(item, int):
            return self.__class__(self.configurations[item])
        return self.__class__(*self.configurations[item])

    @property
    def frames(self) -> np.ndarray:
        """
        Get frames of the ``Trajectory``

        Returns
        -------
        numpy.ndarray
            An ``array`` of `int` specifying the ``frames`` of the
            ``Trajectory``
        """

        return self.data['frame']

    @property
    def times(self) -> np.ndarray:
        """
        Get the times of the ``Trajectory``

        Returns
        -------
        numpy.ndarray
            An ``array`` of `float` specifying the times of the ``Trajectory``
        """

        return self.data['time']

    @property
    def atoms(self) -> 'list[Atom]':
        """
        Get the atoms from the start of the ``Trajectory``

        Returns
        -------
        list
            Atoms from the frame 0 ``Configuration``
        """

        return self.data['configuration'][0].atoms

    @property
    def element_set(self) -> set:
        """
        Get the unique elements from the start of the ``Trajectory``

        Returns
        -------
        set
            Elements from the frame 0 ``Configuration``
        """

        return self.data['configuration'][0].element_set

    @property
    def element_list(self) -> list:
        """
        Get the elements from the start of the ``Trajectory``

        Returns
        -------
        list
            Elements from the frame 0 ``Configuration``
        """

        return self.data['configuration'][0].element_list

    @property
    def configurations(self) -> 'list[Configuration]':
        """
        Get the ``Configuration`` objects of the ``Trajectory``

        Returns
        -------
        list
            A `list` of the ``Configuration``
        """

        return self.data['configuration']

    @property
    def universe(self) -> 'Universe':
        """
        Get the ``Universe`` of the first ``Configuration`` object (the
        ``Universe`` is assumed to be the same for all ``Configuration``s).

        Returns
        -------
        Universe
            The ``Universe`` for the ``Trajectory``
        """

        return self.configurations[0].universe

    @property
    def positions(self) -> np.ndarray:
        """
        Get the positions of the ``Atom`` objects in the ``Trajectory``

        Returns
        -------
        numpy.ndarray
            The ``position`` of every ``Atom`` at every time in the
            ``Trajectory``
        """

        return np.array([position for config in self.configurations
                         for position in config.atom_positions])

    @property
    def velocities(self) -> np.ndarray:
        """
        Get the velocities of the ``Atom`` objects in the ``Trajectory``

        Returns
        -------
        numpy.ndarray
            The ``velocity`` of every ``Atom`` at every time in the
            ``Trajectory``
        """

        return np.array([velocity for config in self.configurations
                         for velocity in config.atom_velocities])

    def filter_by_time(self, start: float, end: 'Optional[float]' = None) -> 'Trajectory':
        """
        Filter the ``Trajectory`` by time

        Parameters
        ----------
        start : float
            The start time for filtering the ``Trajectory``
        end : , optional
            The end time for filtering the ``Trajectory``.  The default is
            `None`, which means the new returned ``Trajectory`` has a single
            time, defined by the ``start``

        Returns
        -------
        Trajectory
            A ``Trajectory`` with ``times`` in half open interval defined by
            ``start`` and ``end``
        """

        if end is None:
            try:
                return self.__class__(*self.configurations[
                    (self.times == start)])
            except IndexError as error:
                raise ValueError("Start is not in self.times") from error
        return self.__class__(*self.configurations[
            (self.times >= start) & (self.times < end)])

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            The number of ``times`` in the ``Trajectory``
        """

        return len(self.times)
