"""Module for ``Configuration`` and ``Trajectory`` classes, and related classes
"""

import weakref

import numpy as np

from MDMC.common.decorators import repr_decorator


class AtomCollection:

    """
    Base class for shared attributes for ``Configurations`` and ``Trajectories``
    """

    @property
    def universe(self):

        """
        Get or set the ``Universe`` in which the ``AtomCollection`` exists

        Returns
        -------
        Universe
            The ``Universe`` in which the ``AtomCollection`` exists, or `None`
            if it has not been set
        """

        try:
            return self._universe()
        except TypeError:
            return None

    @universe.setter
    def universe(self, universe):

        try:
            self._universe = weakref.ref(universe)
        except TypeError:
            self._universe = None

    @property
    def dimensions(self):

        """
        Get the ``dimensions`` of the ``Universe`` in which the
        ``AtomCollection`` exists

        Returns
        -------
        array
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
    *structural_units
        Zero or more ``StructuralUnit`` objects to be added to the
        ``Configuration``
    **settings
        ``universe`` (``Universe``)
            The ``Universe`` of the ``Configuration``

    Attributes
    ----------
    element_set : set
        `set` of the elements in the ``Configuration``
    """

    def __init__(self, *structural_units, **settings):

        try:
            self.universe = settings['universe']
        except KeyError:
            try:
                self.universe = structural_units[0].universe
            except IndexError:
                self.universe = None
        self.data = structural_units
        self.element_set = set(self.element_list)

    @property
    def atom_list(self):

        """
        Get the `list` of ``Atom`` which belong to the ``Configuration``

        Returns
        -------
        list
            A `list` of ``Atom``
        """

        return self.data['atom']

    @property
    def atom_positions(self):

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
    def atom_velocities(self):

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
    def element_list(self):

        """
        Get the `list` of ``Atom.element`` which belong to the ``Configuration``

        Returns
        -------
        list
            A `list` of `str` for the elements
        """

        return [atom.element for atom in self.atom_list]

    @property
    def molecule_list(self):

        """
        Get the `list` of ``Molecule`` which belong to the ``Configuration``

        Returns
        -------
        list
            A `list` of ``Molecule``
        """

        return self.filter_structures(lambda x: x.structure_type == 'Molecule')

    @property
    def data(self):

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
    def data(self, structural_units):

        self.structure_list = []
        self._data = []
        for unit in structural_units:
            self.add_structural_unit(unit)

    def add_structural_unit(self, structural_unit):

        """
        Adds the ``Atom`` objects from a ``StructuralUnit`` to the data

        Parameters
        ----------
        structural_unit : StructuralUnit
            The ``StructuralUnit`` to add
        """

        self.validate_structure(structural_unit)
        self.structure_list.append(structural_unit)
        self._data.extend([atom for atom in structural_unit.atom_list])

    def validate_structure(self, structure):

        """
        Validates the structure by testing that it belongs to the same
        ``Universe`` as the ``Configuration``

        Parameters
        ----------
        structure : StructuralUnit
            The ``StructuralUnit`` to validate

        Raises
        ------
        AssertionError
            If the ``StructuralUnit`` does not belong to the same ``Universe``
            as the ``Configuration``
        """

        # Test that all structural units are from the same universe
        try:
            assert structure.universe is self.universe
        except AssertionError:
            raise AssertionError('Atoms are not all from same universe')

    def __add__(self, configuration):

        """
        Returns
        -------
        Configuration
            New ``Configuration`` from the sum of the ``structure_list`` of the
            two ``Configuation`` objects
        """

        structure_list = self.structure_list + configuration.structure_list

        return self.__class__(*structure_list)

    def __sub__(self, configuration):

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

    def __len__(self):

        """
        Returns
        -------
        int
            The number of ``Atom`` objects in the ``Configuration``
        """

        return len(self.atom_list)

    def __getitem__(self, item):

        """
        Returns
        -------
        numpy.ndarray
            A NumPy ``array`` containing a slice from the data. The same fields
            can be accessed with ``'atom'``, ``'position'``, and ``'velocity'``.
        """

        return self.data[item]

    def filter_structures(self, predicate):

        """
        Filters the `list` of ``StructuralUnits`` using the predicate

        Parameters
        ----------
        predicate : function
            A function which returns a `bool` when passed a ``StructuralUnit``

        Returns
        -------
        list
            A `list` of ``StructuralUnits`` which are `True` for the given
            predicate
        """

        return list(filter(predicate, self.structure_list))

    def filter_atoms(self, predicate):

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

        return list(filter(predicate, self.atom_list))

    def filter_by_element(self, element):

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

    def scale(self, factor, vectors='positions'):

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
        Zero or more ``StructuralUnits``
    """

    def __init__(self, time, *structural_units, **settings):

        super().__init__(*structural_units, **settings)
        self.time = time

    def __add__(self, configuration):

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
    A ``Trajectory`` is a collection of ``TimedConfigurations``

    Parameters
    ----------
    *configurations
        Zero or more ``TimedConfigurations``

    Attributes
    ----------
    configurations : list
        A `list` of ``TimedConfigurations``
    """

    def __init__(self, *configurations):

        self.universe = configurations[0].universe
        self.data = configurations

    @property
    def data(self):

        """
        Get or set the data of the ``Trajectory``

        Returns
        -------
        numpy.ndarray
            An ordered ``array`` of ``frames``, ``times`` (in ``fs``) and
            ``TimedConfigurations``
        """

        return self._data

    @data.setter
    def data(self, configurations):

        self._data = np.array(
            [(i, config.time, config) for
            i, config in enumerate(configurations, 1)],
            dtype = [('frame', 'int64'),
            ('time', 'float64'),
            ('configuration', 'object')])

        for datum in self._data:
            if datum['frame']==1:
                config0 = datum['configuration']
            else:
                self.validate_config(datum['configuration'], validator=config0)

    def validate_config(self, config, validator):

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
            assert len(config.atom_list) == len(validator.atom_list)
        except AssertionError:
            raise AssertionError('Configurations do not contain the same number'
                                 ' of atoms')

    def __getitem__(self, item):

        """
        Indexing and slicing is relative to ``frames``
        """

        if type(item) == int:
            return self.__class__(self.configurations[item])
        else:
            return self.__class__(*self.configurations[item])

    @property
    def frames(self):

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
    def times(self):

        """
        Get the times of the ``Trajectory``

        Returns
        -------
        numpy.ndarray
            An ``array`` of `float` specifying the times of the ``Trajectory``
        """

        return self.data['time']

    @property
    def atoms(self):

        """
        Get the atoms from the start of the ``Trajectory``

        Returns
        -------
        array
            Atoms from the frame 0 ``Configuration``
        """

        return self.data['configuration'][0].atom_list

    @property
    def element_set(self):

        """
        Get the unique elements from the start of the ``Trajectory``

        Returns
        -------
        set
            Elements from the frame 0 ``Configuration``
        """

        return self.data['configuration'][0].element_set

    @property
    def element_list(self):

        """
        Get the elements from the start of the ``Trajectory``

        Returns
        -------
        list
            Elements from the frame 0 ``Configuration``
        """

        return self.data['configuration'][0].element_list

    @property
    def configurations(self):

        """
        Get the ``Configuration`` ovjects of the ``Trajectory``

        Returns
        -------
        list
            A `list` of the ``Configuration``
        """

        return self.data['configuration']

    @property
    def positions(self):

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
    def velocities(self):

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

    def filter_by_time(self, start, end=None):

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
            except IndexError:
                raise ValueError("Start is not in self.times")
        return self.__class__(*self.configurations[
            (self.times >= start) & (self.times < end)])

    def __len__(self):

        """
        Returns
        -------
        int
            The number of ``times`` in the ``Trajectory``
        """

        return len(self.times)
