"""Module in which all structural units are defined.

``Atom`` is the fundamental structural unit in terms of which all others must be
defined.  All shared behaviour is included within the ``StructuralUnit`` base
class."""

from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from copy import deepcopy
from functools import reduce
from itertools import count, permutations
from math import gcd
from types import MethodType
import warnings
import weakref

import numpy as np
from scipy.spatial.transform import Rotation

import MDMC.common.atom_properties as atom_properties
from MDMC.common.decorators import repr_decorator, unit_decorator,\
    unit_decorator_getter
from MDMC.common import units
from MDMC.MD.container import AtomContainer
from MDMC.MD.interaction_functions import Coulomb


@repr_decorator('name', 'ID', 'position', 'velocity', 'parent', 'bounding_box',
                'atom_list')
class StructuralUnit(ABC):

    """Abstract base class for all structural units

    Parameters
    ----------
    position : list, tuple, numpy.ndarray
        A 3 element `list`, `tuple` or ``array`` with the position in units of
        ``Ang``.
    velocity : list, tuple, numpy.ndarray
        A 3 element `list`, `tuple` or ``array`` with the velocity in units of
        ``Ang``.
    name : str
        The name of the structure.

 	Attributes
    ----------
 	ID : int
        A unique identifier for each ``StructuralUnit``.
    universe : Universe
        The ``Universe`` to which the ``StructuralUnit`` belongs.
    name : str
        The name of the structure.
    parent : StructuralUnit
        ``StructuralUnit`` to which this unit belongs, or ``self``
    """

    # ID exists to facilitate a 1 to 1 association with structural units within
    # MD engines.  It may not be required or may only be required for atoms.
    _ID_generator = count(start=1, step=1)

    def __init__(self, position, velocity, name):

        self.ID = self._generate_ID()
        self.position = position
        self.velocity = velocity
        self.name = name
        self.parent = self

    @property
    def position(self):

        """
        Get or set the position of the center of mass of the ``StructuralUnit``
        in ``Ang``

        Returns
        -------
        numpy.ndarray
        """

        return self._position

    @position.setter
    @unit_decorator(unit=units.LENGTH)
    def position(self, position):

        self._position = position

    @property
    def velocity(self):

        """
        Get or set the velocity of the ``StructuralUnit`` in ``Ang/fs``

        Returns
        -------
        numpy.ndarray
        """

        return self._velocity

    @velocity.setter
    @unit_decorator(unit=units.LENGTH / units.TIME)
    def velocity(self, velocity):

        self._velocity = velocity

    @property
    def atom_list(self):

        """
        Get a `list` of all of the `Atom` objects in the structure by
        recursively calling ``atom_list`` for all substructures

        Returns
        -------
        list
            All atoms in the structure
        """

        atom_list = []
        for structure in self._structure_list:
            atom_list.extend(structure.atom_list)
        return atom_list

    @property
    @abstractmethod
    def universe(self):

        """
        Get the ``Universe`` to which the ``StructuralUnit`` belongs

        Returns
        -------
        Universe
            The ``Universe`` to which the ``StructuralUnit`` belongs or `None`
        """

        raise NotImplementedError

    def translate(self, displacement):

        """
        Translate the structural unit by the specified displacement

        Parameters
        ----------
        Displacement : tuple, numpy.ndarray
            A three element tuple or ``array`` of `float`
        """

        self.position = self.position + np.array(displacement)

    @property
    def interactions(self):

        """
        Get a list of the interactions acting on the ``StructuralUnit``

        Returns
        -------
        list
            Interactions acting on the ``StructuralUnit``
        """

        return self.bonded_interactions + self.nonbonded_interactions

    @property
    def bonded_interactions(self):

        """
        Get a list of the bonded interactions acting on the ``StructuralUnit``

        Returns
        -------
        list
            ``BondedInteractions`` acting on the ``StructuralUnit``
        """

        return [pair[0] for pair in self.bonded_interaction_pairs]

    @property
    @abstractmethod
    def nonbonded_interactions(self):

        """
        Get a list of the nonbonded interactions acting on the
        ``StructuralUnit``

        Returns
        -------
        list
            ``NonBondedInteractions`` acting on the ``StructuralUnit``
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def bonded_interaction_pairs(self):

        """
        Get bonded interactions acting on the ``StructuralUnit`` and the other
        atoms to which the atom is bonded


        Returns
        -------
        list
            (``interaction``, ``atoms``) pairs acting on the ``StructuralUnit``,
            where ``atoms`` is a `tuple` of all ``Atom`` objects for that
            specific bonded ``interaction``. At least one of these ``Atom``
            objects belongs to the ``StructuralUnit``

        Example
        -------
        For an O ``Atom`` with two bonds, one to H1 and one to H2::

            >>> print(O.bonded_interaction_pairs)
            [(Bond, (H1, O)), (Bond, (H2, O))]
        """

        raise NotImplementedError

    @property
    def structure_type(self):

        """
        Get the class of the ``StructuralUnit``.

        Returns
        -------
        str
            The name of the class
        """

        return self.__class__.__name__

    @property
    def top_level_structure(self):

        """
        Get the top level structure (i.e. ``StructuralUnit`` which has no
        ``parent``) of the ``StructuralUnit``

        Returns
        -------
        StructuralUnit
            Highest level ``StructuralUnit`` of which it is a member
        """

        if issubclass(type(self.parent), StructuralUnit) \
        and self.parent is not self:
            return self.parent.top_level_structure
        else:
            return self

    def copy(self, position):

        """
        Copies the structural unit and sets the ``position``

        Parameters
        ----------
        position : list, tuple, numpy.ndarray
            3 element `list`, `tuple` or ``array`` of `float` specifying the
            ``position`` of the new ``StructuralUnit``

        Returns
        -------
        StructuralUnit
            A ``StructuralUnit`` of the same type with all non-unique attributes
            copied and a new ``position``
        """

        structural_unit = deepcopy(self)
        structural_unit.position = position
        return structural_unit

    def _generate_ID(self):

        """
        Uses class attribute to generate a unique ``ID`` for each
        ``StructuralUnit``

        Returns
        -------
        int
            Unique `int`
        """

        return next(self._ID_generator)

    def _position_in_parent_CoM_frame(self):

        """
        Get the position in the ``parent`` center of mass frame

        Returns
        -------
        numpy.ndarray
            Position in ``parent`` CoM frame with units of ``Ang``

        Raises
        ------
        AttributeError
            If ``StructuralUnit`` has no ``parent``
        """

        if self.top_level_structure is self:
            raise AttributeError("This structure has no parent")
        else:
            return self.position - self.parent._get_center_of_mass()

    def _added_to_structure(self):

        """
        Method is called if it becomes subunit of another ``StructuralUnit``
        """

        self._position_in_parent = self._position_in_parent_CoM_frame()

    def valid_position(self, position=None):

        """
        Checks if the specified ``position`` is within the bounds of the
        ``StructuralUnit.universe``, if it has one

        Parameters
        ----------
        position : list, tuple, numpy.ndarray
            3 element `list`, `tuple` or ``array`` with units of ``Ang`` or
            `None`. If `None` then the ``position`` of the ``StructuralUnit`` is
            used.

        Returns
        -------
        bool
            `True` if ``position`` is within ``Universe`` or there is no
            associated ``Universe``. `False` if ``StructuralUnit`` has an
            associated ``Universe`` but the ``position`` is not within its
            bounds.

        Raises
        ------
        ValueError
            If ``position`` if undefined
        """

        if position is None:
            position = self.position
        try:
            # (0., 0., 0.) is defined as the origin for all universes
            if (np.any(position < np.array([0., 0., 0])) or
                    np.any(position > self.universe.dimensions)):
                return False
            elif np.any(position == np.float('nan')):
                raise ValueError('position of {0} is undefined'.format(self))
            else:
                return True
        except AttributeError:
            # Not a member of a universe
            return True

    @property
    def bounding_box(self):

        """
        Returns
        -------
        BoundingBox
            Contains the lower and upper extents of the ``Molecule``
        """

        return BoundingBox(self.atom_list)


@repr_decorator('name', 'ID', 'formula', 'position', 'velocity', 'bounding_box',
                'atom_list')
class CompositeStructuralUnit(StructuralUnit, AtomContainer):

    """
    Base class for structural units comprised of more than one ``Atom``
    """

    def __init__(self, position, velocity, name):

        super().__init__(position, velocity, name)
        self.universe = None

    def __deepcopy__(self, memo):

        """
        Copies the ``CompositeStructuralUnit`` and all attributes, except ``ID``
        which is generated

        This will not currently work if the ``CompositeStructuralUnit`` has any
        bonded interactions with atoms external to it (e.g. it may cause issues
        for copying molecules with groups)

        Interactions for ``Atom`` objecys may be reordered with respect to
        initial atoms

        Arguments
        ---------
        memo : dict
            The memoization `dict`
        """

        cls = self.__class__
        unit = cls.__new__(cls)
        memo[id(self)] = unit
        for k, v in self.__dict__.items():
            if k == 'ID':
                setattr(unit, k, self._generate_ID())
            elif (k == '_bonded_interaction_pairs'
                  or k == '_nonbonded_interactions'):
                pass
            elif k == '_structure_list':
                # Seperate structures into atoms and composites
                atoms, composites = [], []
                for s in self._structure_list:
                    (atoms if isinstance(s, Atom) else composites).append(s)

                # Create dict to map from current to new structures. This is
                # used both for creating interactions with correct new atoms,
                # and preserving the structures ordering in unit._structure_list
                struct_map = {}
                for atom in atoms:
                    # Add atom's bonded interactions to memo so that these are
                    # not copied
                    for inter in atom.interactions:
                        if issubclass(type(inter), BondedInteraction):
                            memo[id(inter)] = inter
                    new_atom = deepcopy(atom, memo)
                    struct_map[atom] = new_atom

                # Create interactions
                for inter, pair in self.bonded_interaction_pairs:
                    # try/except accounts for interactions associated with atoms
                    # that are in a composite subunit
                    try:
                        new_pair = [struct_map[atom] for atom in pair]
                        inter.add_atoms(*new_pair)
                    except KeyError:
                        pass

                for composite in composites:
                    new_composite = deepcopy(composite, memo)
                    struct_map[composite] = new_composite

                # List comprehension ensures order of structures in new
                # structure is the same as in original
                setattr(unit, k, [struct_map[s] for s in self._structure_list])
            else:
                setattr(unit, k, deepcopy(v, memo))
        return unit

    def __str__(self):

        """
        Returns
        -------
        str
            The formula and center of mass position of the
            ``CompositeStructuralUnit``
        """

        name = self.name + ' ' if self.name else ''
        return ('{0}{1}  formula: {2}  position: {3}'.format(
            name,
            self.__class__.__name__,
            self.formula,
            self.position))

    @property
    @abstractmethod
    def nonbonded_interactions(self):

        raise NotImplementedError

    @property
    @abstractmethod
    def bonded_interaction_pairs(self):

        raise NotImplementedError

    @property
    def formula(self):

        """
        Get the chemical formula of the ``CompositeStructuralUnit``

        Returns
        -------
        str
            The chemical formula using the Hill system
        """

        return get_reduced_chemical_formula([atom.element for atom
                                             in self.atom_list])

    @property
    def universe(self):

        """
        Get or set the ``Universe`` to which the ``CompositeStructuralUnit``
        belongs

        Returns
        -------
        Universe
            The Universe to which the ``CompositeStructuralUnit`` belongs or
            `None`
        """

        try:
            return self._universe()
        except TypeError:
            return self._universe

    @universe.setter
    def universe(self, value):

        try:
            self._universe = weakref.ref(value)
        except TypeError:
            self._universe = None

        # If top level structure then set the universe of all subunits
        if self.top_level_structure == self:
            for structure in self.structure_list:
                structure.universe = value

    @property
    def structure_list(self):

        """
        Get or set the ``StructuralUnit`` objects that are subunits of this
        ``CompositeStructuralUnit``

        Returns
        -------
        list
            `list` of ``StructuralUnit`` that are subunits of this
            ``CompositeStructuralUnit``
        """

        return self._structure_list

    @structure_list.setter
    def structure_list(self, value):

        self._structure_list = value

    def copy(self, position, rotation=None):
    # pylint:disable=arguments-differ
    # CompositeStructuralUnit's can be rotated, which is meaningless for
    # StructuralUnits in general

        """
        Copies the ``CompositeStructuralUnit`` and all attributes, except ``ID``
        which is generated

        Copying a ``CompositeStructuralUnit`` (e.g. a ``Molecule``) will copy
        all of the ``Atom`` objects within it. All of these atoms will have
        identical bonded and nonbonded interactions to the
        ``CompositeStructuralUnit`` from which they were copied i.e. the
        ``CompositeStructuralUnit`` will be exacltly duplicated. The only
        attributes of the ``CompositeStructuralUnit`` which will differ are the
        ``position`` (which is passed as a Parameter to ``copy``), and the
        ``ID``, which is generated automatically.

        This will not currently work if the ``CompositeStructuralUnit`` has any
        bonded interactions with atoms external to it (e.g. it may cause issues
        for copying molecules with groups)

        Interactions for ``Atom`` objects may be reordered with respect to
        initial atoms

        Parameters
        ----------
        position : list, tuple, numpy.ndarray
            3 element `list`, `tuple` or ``array`` of `float` specifying the
            ``position`` of the new ``StructuralUnit``
        rotation : list, tuple, numpy.ndarray, optional
            3 element `list`, `tuple` or ``array`` of `floats` specifying the
            degrees of anticlockwise rotation around the x, y, and z axes
            respectively. The rotation is centered on the center of mass of the
            ``CompositeStructuralUnit``. The default ``rotation`` is `None`,
            which applies no rotation to the copied ``CompositeStructuralUnit``.

        Returns
        -------
        CompositeStructuralUnit
            A ``CompositeStructuralUnit`` of the same type with all non-unique
            attributes copied and a new ``position``
        """

        composite = super().copy(position)
        if rotation is not None:
            composite.rotate(x=rotation[0], y=rotation[1], z=rotation[2])
        return composite

    def _set_subunit_positions(self):

        """
        Sets the position of all subunits in the global frame in units of
        ``Ang``
        """

        for atom in self.atom_list:
            atom.position = self.position + self._CoM_frame_positions[atom]

    def _calc_CoM(self):

        """
        Returns
        -------
        numpy.ndarray
            Position of the center of mass of the ``CompositeStructuralUnit`` in
            units of ``Ang``
        """

        mass = 0.
        weighted_positions = np.zeros(3)
        for atom in self.atom_list:
            mass += atom.mass
            weighted_positions += (atom.position * atom.mass)
        return weighted_positions / mass

    def _calc_subunit_position_in_CoM_frame(self):

        """
        Calculate the position of all subunits in the
        ``CompositeStructuralUnit`` CoM frame in units of ``Ang``
        """

        self._CoM_frame_positions = {}
        CoM = self._calc_CoM()
        for atom in self.atom_list:
            self._CoM_frame_positions[atom] = atom.position - CoM

    def rotate(self, x=0., y=0., z=0.):

        """
        Rotates the ``CompositeStructuralUnit`` around its center of mass

        In all cases (e.g. x, y and z) the rotation is anticlockwise about the
        specific axis

        Parameters
        ----------
        x : float, optional
            The angle of rotation around the x-axis in ``deg``. The default is
            0.
        y : float, optional
            The angle of rotation around the y-axis in ``deg``. The default is
            0.
        z : float, optional
            The angle of rotation around the z-axis in ``deg``. The default is
            0.
        """

        rotation = Rotation.from_euler('xyz', [x, y, z], degrees=True)
        CoM = self.position
        for atom in self.atom_list:
            atom.position = (CoM
                             + rotation.apply(self._CoM_frame_positions[atom]))


@repr_decorator('name', 'ID', 'element', 'position', 'velocity')
class Atom(StructuralUnit):

    """
    A single atom

    Parameters
    ----------
    element : str
        The atomic element label.
    position : list, tuple, numpy.ndarray, optional
        A 3 element `list`, `tuple` or ``array`` with the position in units of
        ``Ang``. The default is ``(0., 0., 0.)``.
    velocity : list, tuple, numpy.ndarray, optional
        A 3 element `list`, `tuple` or ``array`` with the velocity in units of
        ``Ang``. The default is ``(0., 0., 0.)``.
    charge : float
        The charge of the ``Atom`` in units of elementary charge (``e)``. The
        default is `None`, meaning that a ``Coulomb`` interaction is not applied
        to the ``Atom``.
    **settings
        ``mass`` (`float`)
            The atomic mass in ``amu``. If not provided a lookup table will be
            used.

    Attributes
    ----------
    element : str
        The atomic element label
    """

    def __init__(self, element, position=(0., 0., 0.), velocity=(0., 0., 0.),
                 charge=None, **settings):

        self.universe = None
        super().__init__(position, velocity, name=settings.get('name', element))
        self._nonbonded_interactions = []
        self._bonded_interaction_pairs = []
        self.element = element
        try:
            self.mass = settings['mass']
        except KeyError:
            self.mass = atom_properties.MASS[self.element]
        self._atom_type = settings.get('atom_type', None)
        self.charge = charge

    def __deepcopy__(self, memo):

        """
        Copies the Atom and all attributes, except ``ID`` which is generated

        Interactions are copied but the copied ``Atom`` is substituted for the
        original ``Atom``.  For ``BondedInteractions`` this means that the
        copied ``Atom`` will be bonded to all atoms to which the original
        ``Atom`` is bonded.

        Arguments
        ---------
        memo : dict
            The memoization `dict`
        """

        cls = self.__class__
        atom = cls.__new__(cls)
        memo[id(self)] = atom
        atom._bonded_interaction_pairs = []
        for k, v in self.__dict__.items():
            if k == 'ID':
                setattr(atom, k, self._generate_ID())
            elif k == '_bonded_interaction_pairs':
                self.copy_interactions(atom, memo)
            elif k == '_nonbonded_interactions':
                # All NonBondedInteractions use atom_types so as this will
                # be the same for the new atom then these are automatically
                # applied. The exception is Coulombic interactions initialized
                # with atoms argument. In this case the new atom must be added
                # to the atom_types.
                atom._nonbonded_interactions = []
                for inter in self.nonbonded_interactions:
                    if isinstance(inter, Coulombic):
                        # try/except account for Coulombic interactions
                        # initialized with atom_types
                        try:
                            inter.add_atoms(atom)
                        except AttributeError:
                            atom.add_interaction(inter)
                    else:
                        atom.add_interaction(inter)
            else:
                setattr(atom, k, deepcopy(v, memo))
        return atom

    def __repr__(self):

        """
        Returns
        -------
        tuple
            The ``element``, ``mass``, ``charge``, ``universe``, ``position``,
            ``velocity`` and names of all ``interactions`` that apply to this
            ``Atom``
        """

        return ('{0} atom,'
                '  ID: {1}'
                '  charge: {2},'
                '  interactions: {3}'.format(self.element,
                                             self.ID,
                                             self.charge,
                                             [i.name for i
                                              in self.interactions]))

    def __str__(self):

        """
        Returns
        -------
        str
            The ``element``, ``charge`` and ``position`` of the ``Atom``
        """

        return ('{0} {1}  charge: {2}  position: {3}'.format(
            self.element,
            self.__class__.__name__,
            self.charge,
            self.position))

    @property
    def atom_list(self):

        """
        Get a `list` of the atoms, just consisting of the ``Atom``

        Returns
        -------
        list
            A `list` with a single item, the ``Atom``
        """

        return [self]

    @property
    def universe(self):

        """
        Get the ``Universe`` to which the ``Atomm`` belongs

        Returns
        -------
        Universe
            The ``Universe`` to which the ``Atom`` belongs or `None`
        """

        try:
            return self._universe()
        except TypeError:
            return self._universe

    @universe.setter
    def universe(self, value):

        try:
            self._universe = weakref.ref(value)

            # Update universe for all nonbonded interactions if not previously
            # set
            for inter in self.nonbonded_interactions:
                if not inter.universe:
                    inter.universe = value
        except TypeError:
            self._universe = None

    @property
    def charge(self):

        """
        Get or set the charge in ``e`` if one has been applied to the ``Atom``

        If the ``Atom`` does not have a ``Coulombic`` interaction, setting a
        value of the ``charge`` will create one, and a default ``cutoff`` of
        ``10. Ang`` will be applied

        Returns
        -------
        float
            The charge in units of ``e``, or `None` if no charge has been set

        Raises
        ------
        ValueError
            When the ``Atom`` has more than one ``Coulombic`` interaction
        ValueError
            When the ``Atom`` has more than one parameter; i.e. should only
            have charge as a parameter
        ValueError
            When setting charge to `None` when a ``Coulombic`` interaction
            already exists.
        """

        try:
            num_coul = 0
            value = None
            for interaction in self.interactions:
                if isinstance(interaction, Coulombic):
                    # Check that only one Coulombic interaction exists.
                    num_coul += 1
                    if num_coul > 1:
                        raise ValueError('Atom should not have more than one'
                                         ' Coulombic interaction')
                    # Check that a charge parameter exists.
                    charge_params = 0
                    for param in interaction.params:
                        if param.name == 'charge':
                            charge_params += 1
                            value = param.value
                    if charge_params == 0:
                        raise ValueError('Coulombic interaction does not have a'
                                         ' parameter "charge".')
            return value
        except AttributeError:
            return None

    @charge.setter
    @unit_decorator(unit=units.CHARGE)
    def charge(self, value):

        for inter in self.interactions:
            if isinstance(inter, Coulombic):
                if value is not None:
                    try:
                        for param in inter.params:
                            if param.name == 'charge':
                                param.value = value
                                return
                        raise ValueError('Coulombic interaction does not have'
                                         ' a parameter "charge".')
                    except AttributeError:
                        # creates an interaction function if the Atom's
                        # Coulomb interaction doesn't have one
                        inter.function = Coulomb(value)
                    return
                # else if the charge has value None
                raise ValueError("Can't set charge to None when a"
                                 " Coulombic interaction exists.")
        # Executes if Coulombic interaction doesn't currently exist.
        # Initialises an interaction unless the charge passed is None.
        if value is not None:
            Coulombic(atoms=self, charge=value, cutoff=10.)

    @property
    def mass(self):

        """
        Get or set the atomic mass in ``amu``

        Returns
        -------
        float
            the atomic mass in ``amu``
        """
        return self._mass

    @mass.setter
    @unit_decorator(unit=units.MASS)
    def mass(self, mass):

        self._mass = mass

    @property
    def atom_type(self):

        """
        Get or set the atom type of the ``Atom``

        Returns
        -------
        int
            The atom type

        Raises
        ------
        AttributeError
            The ``atom_type`` cannot be changed once it has been set
        """

        return self._atom_type

    @atom_type.setter
    def atom_type(self, value):

        if self._atom_type:
            raise AttributeError('Can\'t change atom_type once it has been set')
        self._atom_type = value

        # Update atom_types in Coulombic interactions
        for inter in self.nonbonded_interactions:
            if isinstance(inter, Coulombic) and value not in inter.atom_types:
                inter._atom_types.append(value)

    @property
    def nonbonded_interactions(self):

        """
        Get a `list` of the nonbonded interactions acting on the ``Atom``

        Returns
        -------
        list
            ``NonBondedInteractions`` acting on the ``Atom``
        """

        return self._nonbonded_interactions

    @property
    def bonded_interaction_pairs(self):

        """
        Get bonded interactions acting on the ``Atom`` and the other atoms
        to which the ``Atom`` is bonded


        Returns
        -------
        list
            (``interaction``, ``atoms``) pairs acting on the ``Atom``, where
            ``atoms`` is a `tuple` of all `Atom` objects for that specific
            bonded ``interaction``.

        Example
        -------
        For an O ``Atom`` with two bonds, one to H1 and one to H2::

            >>> print(O.bonded_interaction_pairs)
            [(Bond, (H1, O)), (Bond, (H2, O))]
        """

        return self._bonded_interaction_pairs

    def copy(self, position):
    # pylint:disable=useless-super-delegation
    # Docstring specific to Atom
        """
        Copies the ``Atom`` and all attributes, except ``ID`` which is generated

        Copying an ``Atom`` creates an exact duplicate at the specified
        ``position``.  The copied ``Atom`` will have identical bonded and
        nonbonded interactions as the original. For ``BondedInteractions`` this
        means that the copied atom will be bonded to all atoms to which the
        original atom is bonded. The ``ID`` of the copied atom will differ from
        the original, as they are sequentially generated.

        Parameters
        ----------
        position : list, tuple, numpy.ndarray
            A 3 element `list`, `tuple` or ``array`` with the ``position`` of
            the new ``Atom``

        Returns
        -------
        Atom
            A copy of the ``Atom`` with the specified ``position``

        Examples
        --------
        If the following ``Atom`` is copied:

        .. highlight:: python
        .. code-block:: python

            H1 = Atom('H', position=(0., 0., 0.), charge=0.4238)
            H2 = H1.copy(position=(1., 1., 1.))

        then the new ``Atom`` (``H2``) will have no ``BondedInteractions``, but
        will have a ``Coulombic`` interaction, with a ``charge`` of ``0.4238 e``

        If ``H1`` and ``H2`` are then bonded together and a copy is made:

        .. highlight:: python
        .. code-block:: python

            HHbond = Bond((H1, H2))
            H3 = H1.copy(position=(2., 2., 2.))

        then the newest ``Atom`` (``H3``) will have a ``Coulombic`` interaction
        (also with a ``charge`` of ``0.4238 e``), and it will also have a
        ``Bond`` interaction with ``H2`` (as ``H1`` had a ``Bond`` interaction
        with ``H2``).
        """

        return super().copy(position)

    def add_interaction(self, interaction, from_interaction=False):

        """
        Adds an interaction to the ``Atom``

        Parameters
        ----------
        interaction : Interaction
            Any class dervied from ``Interaction``, or any object with base
            class ``Interaction``.  If an ``Interaction`` class is passed then
            it must be a ``NonBondedInteraction`` i.e. only takes a single
            ``Atom`` as an argument. If an ``Interaction`` object is passed then
            this ``Atom`` must be in the ``interaction.atom_list``.
        from_interaction : bool, optional
            Specifies if this method has been called from an ``Interaction``.
            Default is `False`.
        """

        if issubclass(type(interaction), BondedInteraction):
            # The tuple most recently added to interaction.atoms should always
            # contain self
            if from_interaction:
                if not interaction.atoms or not self in interaction.atoms[-1]:
                    raise ValueError('incorrect atom_tuple passed to atom')
            else:
                interaction.add_atoms(self, from_structure=True)
            pair = (interaction, interaction.atoms[-1])
            if pair not in self.bonded_interaction_pairs:
                self._bonded_interaction_pairs.append((interaction, interaction.atoms[-1]))
        else:
            if interaction not in self.nonbonded_interactions:
                self._nonbonded_interactions.append(interaction)

    def copy_interactions(self, atom, memo={}):

        """
        This replicates the interactions from ``self`` for ``Atom``, but with
        ``self`` substituted by ``atom`` in the ``Interaction.atoms``. These
        interactions are added to any that already exist for the ``Atom``.

        Passing the ``memo`` `dict` enables specific interactions to be excluded
        from being copied, duplicating the behaviour of ``__deepcopy__``

        Parameters
        ----------
        atom : Atom
            The ``Atom`` for which ``self.interactions`` are being replicated
        memo : dict, optional
            The memoization `dict`
        """

        # if/else required for deepcopy (where _bonded_interaction_pairs attribute
        # doesn't exist). try/except not valid due to order of operations in
        # add_atoms method.
        if not hasattr(atom, '_bonded_interaction_pairs'):
            atom._bonded_interaction_pairs = []
        for inter, atoms in self.bonded_interaction_pairs:
            if id(inter) not in memo:
                # Maintains order of atoms except with substitution
                new_atoms = [atom if a == self else a for a in atoms]

                # Use add_atoms method to update attribute rather than setattr.
                # This ensures interaction is added to all other atoms as well
                inter.add_atoms(*new_atoms)
                memo[id(inter)] = inter


class _Group(CompositeStructuralUnit):

    """
    Two or more `Atom` objecys that form a subset of a ``Molecule``

    THIS CLASS HAS NOT BEEN IMPLEMENTED AND SO IS CURRENTLY PRIVATE

    Raises
    ------
    NotImplementedError
        THIS CLASS HAS NOT BEEN IMPLEMENTED
    """

    def __init__(self):

        raise NotImplementedError



class Molecule(CompositeStructuralUnit):

    """
    Two or more bonded atoms

    Must be declared with at least 2 ``Atom`` objects.

    Parameters
    ----------
    position : list, tuple, numpy.ndarray, optional
        A 3 element `list`, `tuple` or ``array`` with the position in units of
        ``Ang``. The default is `None`, which sets the position of the
        ``Molecule`` to be equal to the center of mass of the atoms in the
        ``Molecule``.
    velocity : list, tuple, numpy.ndarray, optional
        A 3 element `list`, `tuple` or ``array`` with the velocity in units of
        ``Ang``. The default is ``(0., 0., 0.)``.
    name : str, optional
        The name of the structure. The default is `None`.
    **settings
        ``interactions`` (`list`)
            A `list` of ``Interaction`` acting on atoms within the ``Molecule``.
            The ``interactions`` provides a convenience for declaring
            interactions on atoms when a ``Molecule`` is initialized. It is not
            required and is exactly equivalent to initializing the interactions
            prior to the ``Molecule``.
    """

    def __init__(self, position=None, velocity=(0, 0, 0), name=None,
                 **settings):

        self._structure_list = settings['atoms']
        for structure in self._structure_list:
            structure.parent = self
        self._calc_subunit_position_in_CoM_frame()
        if position is None:
            position = self._calc_CoM()
        super().__init__(position, velocity, name)

    @property
    def position(self):

        """
        Get or set the position of the center of mass of the ``Molecule`` in
        ``Ang``

        Also sets the positions of all subunits

        Returns
        -------
        numpy.ndarray
        """

        return self._position

    @position.setter
    @unit_decorator(unit=units.LENGTH)
    def position(self, position):

        self._position = position
        self._set_subunit_positions()

    @property
    def nonbonded_interactions(self):

        """
        Get a list of the nonbonded interactions acting on the ``Molecule``

        Returns
        -------
        list
            ``NonBondedInteraction`` objects acting on the ``Molecule``
        """

        return [inter for atom in self.atom_list
                for inter in atom.nonbonded_interactions]

    @property
    def bonded_interaction_pairs(self):

        """
        Get bonded interactions acting on the ``Molecule``

        Returns
        -------
        list
            (``interaction``, ``atoms``) pairs acting on the ``Molecule``, where
            ``atoms`` is a `tuple` of all atoms for that specific bonded
            ``interaction``. At least one of these ``atoms`` belongs to the
            ``Molecule``

        Example
        -------
        For an ``O`` ``Atom`` with two bonds, one to ``H1`` and one to ``H2``::

            >>> print(O.bonded_interaction_pairs)
            [(Bond, (H1, O)), (Bond, (H2, O))]
        """

        return list(set([pair for atom in self.atom_list
                         for pair in atom.bonded_interaction_pairs]))

    @property
    @unit_decorator_getter(unit=units.MASS)
    def mass(self):

        """
        Get the molecular mass of the ``Molecule`` in ``amu``

        Returns
        -------
        float
            The molecular mass in ``amu``
        """

        mass = 0
        for atom in self.atom_list:
            mass += atom.mass

        return mass


@repr_decorator('min', 'max', 'volume')
class BoundingBox:

    """
    A box with the minimum and maximum extents of the positions of a collection
    of atoms

    Parameters
    ----------
    atom_list : list
        ``Atom`` objects for which the minimum and maximum extents are
        determined
    """

    def __init__(self, atom_list):

        # Start with arbitrary min and max from the positions of the atoms in
        # the atom list
        self.min = self.max = atom_list.pop().position
        for atom in atom_list:
            self.min = np.minimum(self.min, atom.position)
            self.max = np.maximum(self.max, atom.position)

    @property
    def min(self):

        """
        Get or set the minimum extent of the positions of a collection of atoms

        Returns
        -------
        numpy.ndarray
            The minimum extent in ``Ang``
        """

        return self._min

    @min.setter
    @unit_decorator(unit=units.LENGTH)
    def min(self, value):

        self._min = value

    @property
    def max(self):

        """
        Get or set the maximum extent of the positions of a collection of atoms

        Returns
        -------
        numpy.ndarray
            The maximum extent in ``Ang``
        """

        return self._max

    @max.setter
    @unit_decorator(unit=units.LENGTH)
    def max(self, value):

        self._max = value

    @property
    @unit_decorator_getter(unit=units.LENGTH ** 3)
    def volume(self):

        """
        Get the volume of the bounding box, in units of ``Ang ^ 3``

        Returns
        -------
        float
            The volume of the bounding box
        """

        return abs(np.prod(self.max - self.min))


def filter_atoms(atoms, predicate):

    """
    Filters a list of Atoms with a given predicate

    Parameters
    ----------
    atoms : list
        A `list` of ``Atom``
    predicate : function
        A function that returns a `bool`

    Returns
    -------
    list
        ``Atom`` objects in ``atoms`` which meet the condition of ``predicate``
    """

    return list(filter(predicate, atoms))


def filter_atoms_element(atoms, element):

    """
    Filters a list of atoms based on the atomic element

    Parameters
    ----------
    atoms : list
        A ``list`` of ``Atom``
    element : str
        The atomic element label

    Returns
    -------
    list
        ``Atom`` objects of a specific element
    """

    return list(filter(lambda a: a.element == element, atoms))


def get_reduced_chemical_formula(symbols, factor=None, system='Hill'):

    """
    Get the reduced chemical formula

    Parameters
    ----------
    symbols : list of str
        The chemical formula to be reduced. It is expressed as a `list` of
        elements, with a single element for each atom. Elements are grouped by
        type but not ordered e.g. all ``'O'`` values, then all ``'H'`` values
        etc.
    factor : int, optional
        The factor by which the total number of symbols will be reduced. If
        `None`, the greatest common divisor of the different symbols will be
        used. The default is `None`.
    system : str, optional
        Determines the order of the chemical formula. If ``'Hill'`` the Hill
        system is used to determine the order. If `None`, the order is based on
        the order of ```symbols``. The default is ``'Hill'``.

    Returns
    -------
    str
        The chemical formula corresponding to ``symbols``, except with only
        ``n_atoms``. If `system` is ``'Hill'``, the formula will be ordered as
        per the Hill system, otherwise the formula will be ordered based on the
        order of ``symbols``.

    Example
    -------
    Reducing the formula for four water molecules to a single water molecules:

    .. highlight:: python
    .. code-block:: python

        >>> get_reduced_chemical_formula(['H'] * 8 + ['O'] * 4)
        'H2O'

    Reducing the formula for four water molecules to two water molecules:

    .. highlight:: python
    .. code-block:: python

        >>> get_reduced_chemical_formula(['H'] * 8 + ['O'] * 4, factor=2)
        'H4O2'
    """

    if not factor:
        factor = reduce(gcd, Counter(symbols).values())

    n_symbols = len(symbols)
    if n_symbols % factor != 0:
        raise ValueError('factor ({0}) must be a factor of the number of'
                         ' symbols {1}'.format(factor, n_symbols))

    n_reduced_atoms = n_symbols // factor
    reduced_symbols = symbols[::n_symbols // n_reduced_atoms]

    reduced_symbols_count = OrderedDict()
    # Use keys of OrderedDict to maintain order (and backwards compatibility)
    for symbol in OrderedDict((symbol, None) for symbol
                              in reduced_symbols).keys():
        number = reduced_symbols.count(symbol)
        reduced_symbols_count[symbol] = str(number) if number != 1 else ''

    if system == 'Hill':
        reduced_formula = ''
        if 'C' in reduced_symbols_count:
            reduced_formula = 'C' + reduced_symbols_count.pop('C')
            try:
                reduced_formula += 'H' + reduced_symbols_count.pop('H')
            except KeyError:
                pass

        reduced_formula += ''.join(sorted([symbol + count for symbol, count
                                           in reduced_symbols_count.items()]))
    else:
        reduced_formula = ''.join([symbol + count for symbol, count
                                   in reduced_symbols_count.items()])

    return reduced_formula


@repr_decorator('function')
class Interaction(ABC):

    """
    Base class for interactions, both bonded, non-bonded and constraints

    Each different type of interaction should have an ``Interaction`` object.
    This object contains a `list` of the ``Atom`` (or ```Atom`` pairs, triplets
    or quadruplets,  depending on the type of interaction) for which this
    ``Interaction`` applies. For example, an oxygen ``Coulombic`` interaction
    would contain a `list` of `tuple` where each `tuple` contains a different O
    ``Atom``, and a hydrogen-oxygen ``Bond`` interaction would contain a `list`
    of `tuple` where each `tuple` contains a different H and O pair.
    ``Interaction`` objects can be sliced to return a sublist of the `tuple`.

    When an ``Atom`` is passed to an ``Interaction``, the ``Interaction`` is
    also added to the ``Atom``.

    Parameters
    ----------
    **settings
        ``function`` (`InteractionFunction`)
            A class of interaction function (e.g. ``HarmonicPotential``)
    """

    def __init__(self, **settings):

        """
        Arguments:
        atom_tuples - One or more tuples consisting of one or more Atom objects.
        Each tuple contains all of the atoms involved in a single interaction.
        For example:

        H1 = Atom('H')
        H2 = Atom('H')
        O = Atom('O')

        For non-bonded interactions both of the following are equivalent
        (applying a Dispersion interaction to both H atoms):

        H_dispersive = Dispersion((H1, ), (H2, ))
        H_dispersive = Dispersion(H1, H2)

        For bonded interactions both of the following are equivalent:

        HO_bond = Bond((H1, O))
        HO_bond = Bond(H1, O)

        However when multiple bonds are initialized within a single Bond object
        (e.g. creating a Bond between each H and O for a single water molecule),
        tuples must be used to separate the bonds:

        HO_bond = Bond((H1, O), (H2,O))

        whereas the following is not valid:

        HO_bond = Bond(H1, O, H2, O)
        TypeError: object of type 'Atom' has no len()
        """

        self.function = settings.get('function', None)
        self.name = self.__class__.__name__

    def __deepcopy__(self, memo={}):

        """
        Interactions cannot be copied
        """

        raise AttributeError('Interactions cannot be copied')

    def __copy__(self):

        """
        Interactions cannot be copied
        """

        self.__deepcopy__()

    @property
    @abstractmethod
    def atoms(self):

        """
        Get the atoms on which the ``Interaction`` is applied
        """

        raise NotImplementedError

    @property
    def params(self):

        """
        Get the ``Parameter`` objects belonging to the ``InteractionFunction``
        belonging to the ``Interaction``

        Returns
        -------
        list
            A `list` of the ``Parameter``
        """

        return self.function.params

    @property
    def function(self):

        """
        Get or set the ``InteractionFunction`` of the ``Interaction``

        Returns
        -------
        InteractionFunction
            The interaction function of the ``Interaction``
        """

        return self._function

    @function.setter
    def function(self, value):

        self._function = value

    @property
    def function_name(self):

        """
        Get the name of the ``InteractionFunction`` belonging to the
        ``Interaction``

        Returns
        -------
        str
            The name of the ``InteractionFunction``, or `None` if no
            ``InteractionFunction`` has been set
        """

        try:
            return self.function.name
        except AttributeError:
            return None

    @property
    @abstractmethod
    def universe(self):

        """
        Get the ``Universe`` to which the ``Interaction`` belongs

        Returns
        -------
        Universe
            The ``Universe`` to which the ``Interaction`` belongs or `None`
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def element_list(self):

        """
        Get a `list` of the elements for which the ``Interaction`` applies

        Returns
        -------
        list
            The elements for which the ``Interaction`` applies
        """

        raise NotImplementedError

    def sorted_element_list(self):

        """
        Sort the list of elements for which the ``Interaction`` applies

        Returns
        -------
        list
            The elements for which the ``Interaction`` applies, sorted
            alphabetically
        """

        return sorted(self.element_list())

    def element_tuple(self):

        """
        A `tuple` of elements for which the ``Interaction`` applies

        Returns
        -------
        tuple
            The elements for which the ``Interaction`` applies
        """

        return tuple(self.element_list())

    def _add_interaction_atoms(self, atoms):

        """
        Add the ``Interaction`` to atoms for which the ``Interaction`` has been
        applied

        Parameters
        ----------
        atoms : list
            ``Atom`` objects which have been added to the ``Interaction``
        """

        for atom in atoms:
            atom.add_interaction(self, from_interaction=True)


@repr_decorator('function', 'atom_types', 'cutoff')
class NonBondedInteraction(Interaction):

    """
    Base class for non-bonded interactions

    Parameters
    ----------
    universe : Universe
        The ``Universe`` in which the ``NonBondedInteraction`` exists
    *atom_types
        `int` for each ``atom_type`` for which the ``NonBondedInteraction``
        applies
    **settings
        ``cutoff`` (`float`)
            The distance in ``Ang`` at which the interaction potential is
            truncated
    """

    def __init__(self, universe, *atom_types, **settings):

        self.universe = universe
        if self.universe:
            self.universe.add_nonbonded_interaction(self)
        self.cutoff = settings.get('cutoff')
        super().__init__(**settings)

    @abstractmethod
    def __eq__(self, other):

        raise NotImplementedError

    def __ne__(self, other):

        return not self == other

    def __hash__(self):

        # Simplified version of immutable hash which Python3 produces
        # (marginally less efficient but shouldn't matter)
        return id(self) // 8

    def __str__(self):

        """
        Returns
        -------
        str
            The ``type``, ``atom_types`` and ``cutoff`` of the
            ``NonBondedInteraction``
        """

        return ('{0} interaction  atom_types: {1}  cutoff: {2}'.format(
            self.__class__.__name__,
            self.atom_types,
            self.cutoff))

    @property
    @abstractmethod
    def atom_types(self):

        """
        Get the atom types for which the ``NonBondedInteraction`` applies

        Returns
        -------
        list
            A list of `int` for the ``atom_types``
        """

        raise NotImplementedError

    @property
    def universe(self):

        """
        Get or set the ``Universe`` to which the ``NonBondedInteraction``
        belongs

        Returns
        -------
        Universe
            The ``Universe`` to which the ``NonBondedInteraction`` belongs or
            `None`
        """

        try:
            return self._universe()
        except TypeError:
            return self._universe

    @universe.setter
    def universe(self, value):

        try:
            self._universe = weakref.ref(value)
        except TypeError:
            self._universe = None

    @property
    def cutoff(self):

        """
        Get or set the distance in ``Ang`` at which the interaction potential is
        truncated

        Returns
        -------
        float
            The distance in ``Ang`` of the ``cutoff``
        """
        return self._cutoff

    @cutoff.setter
    @unit_decorator(unit=units.LENGTH)
    def cutoff(self, value):

        self._cutoff = value


class Dispersion(NonBondedInteraction):

    """
    A non-bonded dispersive interaction - either LJ or Buckingham

    Parameters
    ----------
    universe : Universe
        The ``Universe`` in which the ``NonBondedInteraction`` exists
    *atom_types
        `int` for each atom type for which the ``NonBondedInteraction`` applies
    **settings
        ``cutoff`` (`float`)
            The distance in ``Ang`` at which the interaction potential is
            truncated
        ``vdw_tail_correction`` (`bool`)
            Specifies if the tail correction to the energy and pressure should
            be applied. This only affects the simulation dynamics if the
            simulation is being performed with constant pressure.

    Raises
    ------
    TypeError
        ``atom_types`` must be iterable
    ValueError
        ``Dispersion`` should only be specified as existing between pairs of
        ``atom_types``
    TypeError
        Each ``atom_type`` must be `int`
    """

    # Python3 requires subclasses that overwrite __eq__ to explicity inherit
    # __hash__
    __hash__ = NonBondedInteraction.__hash__

    def __init__(self, universe, *atom_types, **settings):

        #Ignore pylint warning for inner function docstring
        #pylint: disable=missing-docstring
        def validate_atom_type_pair(atom_type_pair):
            try:
                atom_type_pair = tuple(sorted(atom_type_pair))
            except TypeError as err:
                raise TypeError('Atom types must be an iterable') from err
            if len(atom_type_pair) != 2:
                raise ValueError('Dispersion interactions should only be'
                                 ' specified as existing between pairs of'
                                 ' atom types')
            if not all([isinstance(atom_type, (int, np.integer)) for atom_type
                        in atom_type_pair]):
                raise TypeError('Each atom type must be int')
            return atom_type_pair

        # Remove duplicates
        self._atom_types = tuple(validate_atom_type_pair(atp) for atp
                                 in atom_types)
        super().__init__(universe, **settings)
        # Add interactions to all atoms
        for atom_type_pair in self.atoms:
            for atoms in atom_type_pair:
                for atom in atoms:
                    atom.add_interaction(self)

        self.vdw_tail_correction = settings.get('vdw_tail_correction', False)

    def __eq__(self, other):

        return other.atom_types == self.atom_types and isinstance(other,
                                                                  type(self))

    @property
    def atom_types(self):

        return tuple(sorted(self._atom_types))

    @property
    def atoms(self):

        """
        Get the atoms on which the ``Dispersion`` is applied

        Returns
        -------
        list
            A `list` of two `tuple`, where each `tuple` contains a `list` of
            `Atom`. Every ``Atom`` in the first `tuple` has a dispersion
            interaction with every ``Atom`` in the second `tuple` (excluding
            self interactions). This is the complete list of possible dispersion
            interactions, i.e. it is only exactly correct if no cutoff has been
            specified.
        """

        return [map(lambda x: self.universe.atom_types[x], tpl) for tpl
                in self.atom_types]

    def element_list(self):

        """
        Get a list of the elements for which the ``Interaction`` applies

        Returns
        -------
        list
            The elements for which the ``Interaction`` applies
        """

        # Each value in universe.atom_types dictionary contain list of atoms
        # with same elements, so use index 0
        # This is determined for all atom types in Dispersion interaction
        return [self.universe.atom_types[atom_type][0].element
                for tpl in self.atom_types
                for atom_type in tpl]


class Coulombic(NonBondedInteraction):

    """
    A non-bonded coulombic interaction - either normal or modified Coulomb

    Parameters
    ----------
    universe : Universe, optional
        The ``Universe`` in which the ``Coulombic`` exists. Default is `None`.
        Must be passed as a parameter if ``atom_types`` if passed.
    **settings
        ``charge`` (`float`)
            The charge parameter of the ``Coulombic`` interaction, in units of
            ``e``. If this argument is passed, the ``interaction_function`` of
            this ``Coulombic`` is set to ``Coulomb`` with this `float` as its
            ``Parameter``. Passing ``charge`` will overwrite any other
            ``interaction_functions`` that are set, i.e. it makes ``function``
            parameter redundant
        ``atoms`` (`list`)
            ``Atom`` objects to which the ``Coulombic`` applies. If specifying
            the ``atoms``, ``universe`` does not need to be passed as a
            parameter.



        atom_types : list of int
            int for each atom_type for which the NonBondedInteraction applies.
            If specifying the atom_types, the universe must be passed as a
            parameter and the atoms for which the atom_types are specified must
            exist in Universe. See the example above in the 'charge' section.

    Raises
    ------
    TypeError
        If one or more atom_types are passed but no universe is passed
    TypeError
        If neither atom_types or atoms have been passed
    TypeError
        If both atom_types and atoms have been passed

    Warns
    -----
    UserWarning
        If a charge is set when the Atom has no Coulombic interaction,
        resulting in the initialization of a Coulomb interaction function.
        Warning only raised in the first instance of triggering behaviour.

    Examples
    --------
    Upon initializing an ``Atom`` object and adding it to a ``Universe``:

    .. highlight:: python
    .. code-block:: python

        O = Atom('O', atom_type=1)
        universe = Universe(10.0)
        universe.add_structural_unit('O')

    The following initializations of Coulombic are equivalent:

    .. highlight:: python
    .. code-block:: python

        O_coulombic = Coulombic(universe, atom_types=[O.atom_type],
                                charge=-0.84)
        O_coulombic = Coulombic(universe, atom_types=[O.atom_type],
                                function=Coulomb(-0.84))

    If ``atoms`` is passed then a ``Universe`` does not need to be passed:

    .. highlight:: python
    .. code-block:: python

        O_coulombic = Coulombic(atoms=[O], charge=-0.84)
    """

    # Python3 requires subclasses thay overwrite __eq__ to explicity inherit
    # __hash__
    __hash__ = NonBondedInteraction.__hash__

    def __init__(self, universe=None, **settings):

        try:
            atom_types = settings['atom_types']
            if settings.get('atoms'):
                raise TypeError('Cannot pass both atoms and atom_types '
                                'as parameters.')
            if isinstance(atom_types, (int, np.integer)):
                # Account for init argument atom_types=atom_type
                # rather than atom_types=[atom_type]
                atom_types = [atom_types]
            if not universe:
                raise TypeError('Coulombic requires a universe when '
                                'atom_types are passed')

            self.add_atom_types = MethodType(_add_atom_types, self)
            self._atom_types = atom_types
            self._atoms = [atom for atom_type in self.atom_types
                           for atom in universe.atom_types[atom_type]]
            super().__init__(universe, **settings)
            # Add interaction to atoms
            for atom in self.atoms:
                atom.add_interaction(self)
        except KeyError:
            self.add_atoms = MethodType(_add_atoms, self)
            try:
                atoms = settings['atoms']
            except KeyError:
                raise TypeError('Coulombic takes either atom_types or atoms '
                                'as parameters')
            # Account for init argument atoms=atom rather than atoms=[atom]
            if isinstance(atoms, Atom):
                atoms = [atoms]
            self._atoms = []
            self._atom_types = []
            self.add_atoms(*atoms)

            # Assumes all atoms are in the same universe (or None)
            universe = self.atoms[0].universe
            super().__init__(universe, **settings)

        charge = settings.get('charge')
        if charge is not None:
            # Initializes a Coulomb interaction function with charge and units
            # and assigns it to self.function
            self.function = Coulomb(charge)
            warnings.warn(UserWarning('Coulombic interaction for the Atom '
                                      'object initialized with the Coulomb '
                                      'interaction function.'))

    def __len__(self):

        """
        Returns
        -------
        int
            The number of interactions of this type that have been set
        """

        return len(self.atoms)

    def __getitem__(self, key):

        """
        Returns
        -------
        tuple
            The `tuple` of atoms at the specified index in ``atoms``.
        """

        return self.atoms[key]

    def __eq__(self, other):

        return (isinstance(other, type(self))
                and (sorted(self.atom_types, key=id)
                     == sorted(other.atom_types, key=id))
                and sorted(self.atoms, key=id) == sorted(other.atoms, key=id))

    @property
    def atoms(self):

        """
        Get the atoms on which the ``Coulombic`` interaction is applied

        Returns
        -------
        list
            A `list` of ``Atom`` on which the ``Coulombic`` is applied
        """

        return self._atoms

    @property
    def atom_types(self):

        """
        Get the atom types for which the ``Coulombic`` applies

        Returns
        -------
        list
            All atom types to which the ``Coulombic`` applies. If the
            interaction was initialized with ``atoms``, all ``atom_types``
            of the ``atoms`` to which the ``Coulombic`` was applied are
            returned; HOWEVER THE COULOMBIC INTERACTION IS NOT APPLIED TO ALL
            ATOMS OF THESE ``atom_types``, ONLY THE ATOMS IN ``self.atoms``
        """

        return self._atom_types

    def element_list(self):

        """
        Get a list of the elements for which the ``Coulombic`` interaction applies

        Returns
        -------
        list
            The elements for which the ``Coulombic`` interaction applies
        """

        return list(set([atom.element for atom in self._atoms]
                        + [self.universe.atom_types[atom_type][0].element for
                           atom_type in self.atom_types]))


def _add_atom_types(self, *atom_types):

    """
    Function for dynamically creating an ``add_atom_types`` method in
    ``Coulombic``

    Parameters
    ----------
    atom_types : list
        One or more `int` specifying ``atom_types`` that exist in ``universe``
        of the ``Coulombic``
    """

    self._atom_types.append(*atom_types)


def _add_atoms(self, *atoms):

    """
    Function for dynamically creating an ``add_atoms`` method in ``Coulombic``

    Adds ``*atoms`` to ``Coulombic`` and adds ``Coulombic`` to
    ``atoms.nonbonded_interactions``

    Parameters
    ----------
    atoms : list
        list of ``Atom``
    """

    for atom in atoms:
        # Add atom to interaction
        self._atoms.append(atom)
        # Add interaction to atom
        atom.add_interaction(self, from_interaction=True)
        # Add atom_type to interaction.atom_types
        if atom.atom_type and atom.atom_type not in self.atom_types:
            self._atom_types.append(atom.atom_type)


@repr_decorator('function', 'n_atoms')
class BondedInteraction(Interaction):

    """
    Base class for bonded interactions

    Parameters
    ----------
    atom_tuples : list
        A `list` of `tuple`. Each `tuple` contains ``Atom`` objects which are
        bonded together. For three or more ``Atom`` objects, the order of the
        ``Atom`` objecys within each `tuple` is important.
    **settings
        n_atoms : int
            The number of atoms to which this ``BondedInteraction`` applies, for
            example 2 for a ``Bond``.

    Examples
    --------
    For a single bonded interactions which applies to ``H1``, ``O1``, and
    ``H2``:

    .. highlight:: python
    .. code-block:: python

        BondedInteraction(H1, O1, H2)

    For two bonded interactions of the same ``BondedInteraction`` type, one
    applied to ``H1``, ``O1`` and ``H2`` ``Atom`` objects, and the other applied
    to ``H3``, ``O2`` and ``H4`` ``Atom`` objects:

    .. highlight:: python
    .. code-block:: python

        BondedInteraction((H1, O1, H2), (H3, O2, H4))

    Whereas the above examples are both specifying a H-O-H ordered
    ``BondedInteraction``, the following specifies a H-H-O
    ``BondedInteraction``:

    .. highlight:: python
    .. code-block:: python

        BondAngle(H1, H2, O)
    """

    def __init__(self, *atom_tuples, **settings):

        if atom_tuples and isinstance(atom_tuples[0], Atom):
            atom_tuples = (atom_tuples, )
        if settings.get('n_atoms'):
            # This ensures that BondedInteractions can also be __init__ with 0
            # atoms
            for tpl in atom_tuples:
                self._validate_atoms(tpl, settings.get('n_atoms'))
        self.atoms = list(atom_tuples)
        super().__init__(**settings)

    def __len__(self):

        """
        Returns
        -------
        int
            The number of interactions of this type that have been set
        """

        return len(self.atoms)

    def __getitem__(self, key):

        """
        Returns
        -------
        tuple
            The `tuple` of ``Atom`` at the specified index.  For a single index
            (as opposed to a slice) this is a group of atoms which are bonded
            together.
        """

        return self.atoms[key]

    def __str__(self):

        """
        Returns
        -------
        str
            The type, and number of atoms of the ``BondedInteraction``
        """

        return ('{0} interaction applied to {1} atom tuples'.format(
            self.__class__.__name__,
            len(self.atoms)))

    @property
    def atoms(self):

        """
        Get or set the atoms on which the ``Coulombic`` interaction is applied

        Returns
        -------
        list
            A `list` of `tuple` containing one or more ``Atom``. Each `tuple`
            contains all of the atoms involved in one example of the
            interaction. For example a ``BondAngle`` interaction each `tuple`
            would contain 3 or 4 atoms.

        Raises
        ------
        TypeError
            If a `list` of `tuple` is not set
        """

        return self._atoms

    @atoms.setter
    def atoms(self, atom_tuples):

        # Check for duplicate tuples in list
        self._check_duplicates(atom_tuples)
        # Check for duplicate atoms in each tuple
        try:
            for tpl in atom_tuples:
                self._check_duplicates(tpl)
        # try/except accounts for single atom passed rather than (atom,) tuple
        # e.g. if atom_tuples = [atom] instead of atom_tuples = [(atom,)]
        except TypeError:
            if len(atom_tuples) == 1 and isinstance(atom_tuples[0], Atom):
                atom_tuples = [(atom_tuples[0],)]
            else:
                raise TypeError('atom_tuples must be [(atom, ...), ...]')

        # Only assign interaction to atoms after these validation steps
        self._atoms = []
        for tpl in atom_tuples:
            # Each tuple is appended individually so that it can be easily added
            # to ._bonded_interaction_pairs for every atom in the tuple
            self._atoms.append(tpl)
            self._add_interaction_atoms(tpl)

            # Add interaction to Universe (pass if no universe exists)
            try:
                self._add_to_universe(self.universe, tpl)
            except AttributeError:
                pass

    @property
    def universe(self):

        """
        Get the ``Universe`` to which the ``BondedInteraction`` belongs

        Returns
        -------
        Universe
            The ``Universe`` to which the ``BondedInteraction`` belongs or
            `None`
        """

        try:
            return self.atoms[0][0].universe
        except IndexError:
            return None

    def element_list(self):

        """
        Get a `list` of the elements for which the ``BondedInteraction`` applies

        Returns
        -------
        list
            The elements for which the ``BondedInteraction`` applies
        """

        try:
            # Each tuple should contain the same elements, so first tuple's used
            return [atom.element for atom in self.atoms[0]]
        except (AttributeError, IndexError):
            return None

    def _validate_atoms(self, atoms, n_atoms):

        """
        Validates that the correct number of atoms have been passed to the
        interaction

        Parameters
        ----------
        atoms : list
            A `list` of ``Atom`` to validate
        n_atoms : int
            The expected number of ``Atom`` objects in ``atoms``

        Raises
        ------
        TypeError
            If the number of ``Atom`` objects in ``atoms`` is not equal to
            ``n_atoms``
        """

        if len(atoms) not in n_atoms:
            raise TypeError("This interaction only accepts {0} atoms".format(
                n_atoms))

    def add_atoms(self, *atoms, **settings):

        """
        Add atoms which are all involved in one example of this interaction

        Parameters
        ----------
        *atoms
            one or more ``Atom`` objects
        **settings
            ``from_structure`` (`bool`)
                If ``add_atoms`` has been called from a ``StructuralUnit``

        Raises
        ------
        ValueError
            If this ``BondedInteraction`` has already been applied to one or
            more of the ``atoms``
        """

        self._check_duplicates(atoms)
        if atoms in self.atoms:
            raise ValueError('This interaction has already been applied to this'
                             ' atom(s)')

        self._atoms.append(atoms)
        from_structure = settings.get('from_structure', False)
        if not from_structure:
            for atom in atoms:
                atom.add_interaction(self, from_interaction=True)

        if self.universe:
            self._add_to_universe(self.universe, atoms)

    def _check_duplicates(self, structs):

        """
        Checks for duplicates ``StructuralUnit``

        Parameters
        ----------
        structs : list
            A `list` of ``StructuralUnit``
        err_msg : str
            A `str` to provide as an error message if there is a duplicate
            ``StructuralUnit``

        Raises
        ------
        ValueError
            If there is a duplicate ``StructuralUnit``
        """

        err_msg = ('Each tuple in the list of atom tuples must be unique, and'
                   ' each atom in a tuple must be unique')

        # Check for duplicates (or reverse duplicates)
        try:
            equivalent_structs = self._get_equivalent_structures(structs)
        except TypeError:
            equivalent_structs = structs
        if len(set(equivalent_structs)) != len(equivalent_structs):
            raise ValueError(err_msg)

    # _get_equivalent_structures is a method because of the override in
    # DihedralAngle
    #pylint: disable=R0201
    def _get_equivalent_structures(self, structs):

        """
        Returns
        -------
        list
            `list` of `tuple` of ``Atom`` orderings which are equivalent
        """

        return structs + [tuple(reversed(atom_tuple)) for atom_tuple in structs]

    def _add_to_universe(self, universe, tpl):

        """
        Adds interaction and atom tuple to ``universe``

        Parameters
        ----------
        universe : Universe
            The ``Universe`` to which to add the ``Interaction`` and `tpl`
        tpl : tuple
            A `tuple` of ``Atom``
        """

        universe.add_bonded_interaction_pairs((self, tpl))

@repr_decorator('constrained')
class Constrainable:

    """
    A mixin class enabling classes inheriting from ``BondedInteraction`` to be
    constrained

    These constraints are then applied by a constraint algorithm (e.g. SHAKE),
    which is specified in the ``Universe`` to which the ``BondedInteraction``
    belongs.

    Parameters
    ----------
    atom_tuples : list
        A `list` of `tuple`. Each `tuple` contains ``Atom`` objects which are
        bonded together. For three or more ``Atom`` objects, the order of the
        ``Atom`` objects within each `tuple` is important.
    **settings

    Attributes
    ----------
    constrained : bool
        Specifying whether the object is constrained
    """

    def __init__(self, *atom_tuples, **settings):

        self.constrained = settings.get('constrained', False)
        super().__init__(*atom_tuples, **settings)


@repr_decorator('function', 'constrained')
class Bond(Constrainable, BondedInteraction):

    """
    A bond between any two atoms. Requires exactly two atoms in each
    ``atom_tuple``.

    Parameters
    ----------
    atom_tuples : list
        A `list` of `tuple`. Each `tuple` contains ``Atom`` which are bonded
        together.
    **settings
    """

    def __init__(self, *atom_tuples, **settings):

        settings['n_atoms'] = (2, )
        super().__init__(*atom_tuples, **settings)


@repr_decorator('function', 'constrained')
class BondAngle(Constrainable, BondedInteraction):

    """
    A bond angle between any two bonds

    Requires three ``Atom`` objects (rotation around central atom) in each
    ``atom_tuple``. The atoms are ordered ``i``, ``j``, ``k``, where ``j`` is
    the central atom.  So:

    .. highlight:: python
    .. code-block:: python

        BondAngle(i, j, k) == BondAngle(k, j, i)

    Parameters
    ----------
    atom_tuples : list
        A `list` of `tuple`. Each `tuple` contains ``Atom`` which are bonded
        together. For three or more ``Atom`` objects, the order of the ``Atom``
        objects within each `tuple` is important.
    **settings
    """

    def __init__(self, *atom_tuples, **settings):

        settings['n_atoms'] = (3, )
        super().__init__(*atom_tuples, **settings)


@repr_decorator('function', 'improper')
class DihedralAngle(BondedInteraction):

    """
    A dihedral angle between any two sets of three atoms, ``ijk`` and ``jkl``.

    Dihedral angles can be both proper and improper, where the angle between the
    two planes of ``ijk`` and ``jkl`` is fixed for improper dihedrals.

    The atoms of a proper ``DihedralAngle`` are ordered ``i``, ``j``, ``k``,
    ``l``, where ``j`` and ``k`` are the two central atoms.  So:

    .. highlight:: python
    .. code-block:: python

        DihedralAngle(i, j, k, l) == DihedralAngle(l, k, j, i)

    The atoms of an improper ``DihedralAngle`` are ordered ``i``, ``j``, ``k``,
    ``l``, where ``i`` is the central atom to which ``j``, ``k``, and ``l`` are
    all connected. So:

    .. highlight:: python
    .. code-block:: python

        (DihedralAngle(i, j, k, l, improper=True)
        == DihedralAngle(i, j, l, k, improper=True)
        == DihedralAngle(i, l, k, j, improper=True)
        == DihedralAngle(i, l, j, k, improper=True))
        == DihedralAngle(i, k, j, l, improper=True))
        == DihedralAngle(i, k, l, j, improper=True))

    Parameters
    ----------
    atom_tuples : list
        A `list` of `tuple`. Each `tuple` contains four `Atom` objects which are
        bonded together by the ``DihedralAngle``, in the order specified.
    **settings
        ``improper`` (`bool`)
            Whether the ``DihedralAngle`` is improper or not.

    Attributes
    ----------
    improper : bool
        Whether the ``DihedralAngle`` is improper or not, which affects the
        ``InteractionFunction`` which can be set for this ``DihedralAngle``. By
        default this is set to `False` i.e. the interaction is a proper
        dihedral.
    """

    def __init__(self, *atom_tuples, **settings):

        settings['n_atoms'] = (4, )
        self.improper = settings.get('improper', False)
        super().__init__(*atom_tuples, **settings)

    def _get_equivalent_structures(self, structs):

        """
        Returns
        -------
        list
            `list` of `tuple` of ``Atom`` orderings which are equivalent
        """

        # Improper dihedrals are equivalent if they have the same first
        # (central) atom, and any permutation of the other three atoms
        if self.improper:
            equivalent = []
            for atom_tuple in structs:
                equivalent += [(atom_tuple[0], ) + permutation for permutation
                               in permutations(atom_tuple[1:])]
            return equivalent
        # Proper dihedrals are equivalent if they are reversed (as with Bond and
        # BondAngle)
        return super()._get_equivalent_structures(structs)
