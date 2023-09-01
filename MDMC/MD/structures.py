"""Module in which all structural units are defined.

``Atom`` is the fundamental structural unit in terms of which all others must be
defined.  All shared behaviour is included within the ``Structure`` base
class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from copy import deepcopy
from functools import lru_cache, reduce
from itertools import count
import logging
from math import gcd
from typing import Callable, Union, TYPE_CHECKING
import warnings
import weakref

import numpy as np
from scipy.spatial.transform import Rotation

from MDMC.common import atom_properties
from MDMC.MD.interactions import Coulombic, BondedInteraction
from MDMC.common.decorators import repr_decorator, unit_decorator,\
    unit_decorator_getter
from MDMC.common import units
from MDMC.MD.container import AtomContainer
from MDMC.MD.interaction_functions import Coulomb

if TYPE_CHECKING:
    from MDMC.MD.simulation import Universe
    from MDMC.MD.interactions import Interaction, NonBondedInteraction


LOGGER = logging.getLogger(__name__)


@repr_decorator('name', 'ID', 'position', 'velocity', 'parent', 'bounding_box',
                'atoms')
class Structure(ABC):
    # pylint: disable=no-member
    # to avoid errors with MD and _structure_list
    """Abstract base class for all structural units

    Parameters
    ----------
    position : list, tuple, numpy.ndarray
        A 3 element `list`, `tuple` or ``array`` with the position in units of
        ``Ang``.
    velocity : list, tuple, numpy.ndarray
        A 3 element `list`, `tuple` or ``array`` with the velocity in units of
        ``Ang / fs``.
    name : str
        The name of the structure.

        Attributes
    ----------
        ID : int
        A unique identifier for each ``Structure``.
    universe : Universe
        The ``Universe`` to which the ``Structure`` belongs.
    name : str
        The name of the structure.
    parent : Structure
        ``Structure`` to which this unit belongs, or ``self``
    """

    # ID exists to facilitate a 1 to 1 association with structural units within
    # MD engines.  It may not be required or may only be required for atoms.
    _ID_generator = count(start=1, step=1)

    def __init__(self,
                 position: Union['list[float]', 'tuple[float]', np.ndarray],
                 velocity: Union['list[float]', 'tuple[float]', np.ndarray],
                 name: str):

        self.ID = self._generate_ID()
        self.position = position
        self.velocity = velocity
        self.name = name
        self.parent = self
        self._position_in_parent = None

        LOGGER.info('%s created: {ID:%s, name:%s, position:%s}',
                    self.__class__,
                    self.ID,
                    self.name,
                    self.position)

    @property
    def position(self) -> np.ndarray:
        """
        Get or set the position of the center of mass of the ``Structure``
        in ``Ang``

        Returns
        -------
        numpy.ndarray
        """

        return self._position

    @position.setter
    @unit_decorator(unit=units.LENGTH)
    def position(self, position: np.ndarray) -> None:

        self._position = position

    @property
    def velocity(self) -> np.ndarray:
        """
        Get or set the velocity of the ``Structure`` in ``Ang/fs``

        Returns
        -------
        numpy.ndarray
        """

        return self._velocity

    @velocity.setter
    @unit_decorator(unit=units.LENGTH / units.TIME)
    def velocity(self, velocity: np.ndarray) -> None:

        self._velocity = velocity

    @property
    def atoms(self) -> 'list[Atom]':
        """
        Get a `list` of all of the `Atom` objects in the structure by
        recursively calling ``atoms`` for all substructures

        Returns
        -------
        list
            All atoms in the structure
        """

        return [atom for structure in self._structure_list
                for atom in structure.atoms]

    @property
    @abstractmethod
    def universe(self) -> Union['Universe', None]:
        """
        Get the ``Universe`` to which the ``Structure`` belongs

        Returns
        -------
        Universe
            The ``Universe`` to which the ``Structure`` belongs or `None`
        """

        raise NotImplementedError

    def translate(self, displacement: Union[tuple, np.ndarray]) -> None:
        """
        Translate the structural unit by the specified displacement

        Parameters
        ----------
        Displacement : tuple, numpy.ndarray
            A three element tuple or ``array`` of `float`
        """

        self.position = self.position + np.array(displacement)

    @property
    def interactions(self) -> 'list[Interaction]':
        """
        Get a list of the interactions acting on the ``Structure``

        Returns
        -------
        list
            Interactions acting on the ``Structure``
        """

        return self.bonded_interactions + self.nonbonded_interactions

    @property
    def bonded_interactions(self) -> 'list[BondedInteraction]':
        """
        Get a list of the bonded interactions acting on the ``Structure``

        Returns
        -------
        list
            ``BondedInteractions`` acting on the ``Structure``
        """

        return [pair[0] for pair in self.bonded_interaction_pairs]

    @property
    @abstractmethod
    def nonbonded_interactions(self) -> 'list[NonBondedInteraction]':
        """
        Get a list of the nonbonded interactions acting on the
        ``Structure``

        Returns
        -------
        list
            ``NonBondedInteractions`` acting on the ``Structure``
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def bonded_interaction_pairs(self) -> 'list[tuple[Interaction, tuple[Atom]]]':
        """
        Get bonded interactions acting on the ``Structure`` and the other
        atoms to which the atom is bonded


        Returns
        -------
        list
            (``interaction``, ``atoms``) pairs acting on the ``Structure``,
            where ``atoms`` is a `tuple` of all ``Atom`` objects for that
            specific bonded ``interaction``. At least one of these ``Atom``
            objects belongs to the ``Structure``

        Example
        -------
        For an O ``Atom`` with two bonds, one to H1 and one to H2::

            >>> print(O.bonded_interaction_pairs)
            [(Bond, (H1, O)), (Bond, (H2, O))]
        """

        raise NotImplementedError

    @property
    def structure_type(self) -> str:
        """
        Get the class of the ``Structure``.

        Returns
        -------
        str
            The name of the class
        """

        return self.__class__.__name__

    @property
    def top_level_structure(self) -> Structure:
        """
        Get the top level structure (i.e. ``Structure`` which has no
        ``parent``) of the ``Structure``

        Returns
        -------
        Structure
            Highest level ``Structure`` of which it is a member
        """

        if issubclass(type(self.parent), Structure) \
           and self.parent is not self:
            return self.parent.top_level_structure
        return self

    def copy(self, position: Union['list[float]', 'tuple[float]', np.ndarray]) -> Structure:

        """
        Copies the structural unit and sets the ``position``

        Parameters
        ----------
        position : list, tuple, numpy.ndarray
            3 element `list`, `tuple` or ``array`` of `float` specifying the
            ``position`` of the new ``Structure``

        Returns
        -------
        Structure
            A ``Structure`` of the same type with all non-unique attributes
            copied and a new ``position``
        """

        structures = deepcopy(self)
        structures.position = position
        return structures

    def _generate_ID(self) -> int:
        """
        Uses class attribute to generate a unique ``ID`` for each
        ``Structure``

        Returns
        -------
        int
            Unique `int`
        """

        return next(self._ID_generator)

    def _position_in_parent_CoM_frame(self) -> np.ndarray:
        """
        Get the position in the ``parent`` center of mass frame

        Returns
        -------
        numpy.ndarray
            Position in ``parent`` CoM frame with units of ``Ang``

        Raises
        ------
        AttributeError
            If ``Structure`` has no ``parent``
        """

        if self.top_level_structure is self:
            raise AttributeError("This structure has no parent")
        return self.position - self.parent.get_center_of_mass()

    def _added_to_structure(self) -> None:
        """
        Method is called if it becomes subunit of another ``Structure``
        """

        self._position_in_parent = self._position_in_parent_CoM_frame()

    def valid_position(self,
                       position: Union['list[float]', 'tuple[float]', np.ndarray] = None) -> bool:
        """
        Checks if the specified ``position`` is within the bounds of the
        ``Structure.universe``, if it has one

        Parameters
        ----------
        position : list, tuple, numpy.ndarray
            3 element `list`, `tuple` or ``array`` with units of ``Ang`` or
            `None`. If `None` then the ``position`` of the ``Structure`` is
            used.

        Returns
        -------
        bool
            `True` if ``position`` is within ``Universe`` or there is no
            associated ``Universe``. `False` if ``Structure`` has an
            associated ``Universe`` but the ``position`` is not within its
            bounds.

        Raises
        ------
        ValueError
            If ``position`` if undefined
        """
        # pylint: disable=nan-comparison

        # because math.isnan doesn't work here for some reason

        if position is None:
            position = self.position
        try:
            # (0., 0., 0.) is defined as the origin for all universes
            if (np.any(position < np.array([0., 0., 0])) or
                    np.any(position > self.universe.dimensions)):
                return False
            if np.any(position == float('nan')):
                raise ValueError('position of {0} is undefined'.format(self))
            return True
        except AttributeError:
            # Not a member of a universe
            return True

    @property
    def bounding_box(self) -> BoundingBox:
        """
        Returns
        -------
        BoundingBox
            Contains the lower and upper extents of the ``Molecule``
        """

        return BoundingBox(self.atoms)

    @abstractmethod
    def is_equivalent(self, structure: Structure) -> bool:
        """
        Checks the passed ``Structure`` against `self` for equivalence in
        terms of the force field application, namely that the following are the
        same:
          - Element or chemical formula
          - Mass
          - Charge
          - Bonded interactions
          - Non bonded interactions


        Returns
        -------
        bool
            Whether the two are equivalent
        """

        raise NotImplementedError


@repr_decorator('name', 'ID', 'formula', 'position', 'velocity', 'bounding_box',
                'atoms')
class CompositeStructure(Structure, AtomContainer):
    """
    Base class for structural units comprised of more than one ``Atom``
    """

    def __init__(self,
                 position: Union['list[float]', 'tuple[float]', np.ndarray],
                 velocity: Union['list[float]', 'tuple[float]', np.ndarray],
                 name: str):

        super().__init__(position, velocity, name)
        self.universe = None

    def __deepcopy__(self, memo: dict) -> CompositeStructure:
        """
        Copies the ``CompositeStructure`` and all attributes, except ``ID``
        which is generated

        This will not currently work if the ``CompositeStructure`` has any
        bonded interactions with atoms external to it (e.g. it may cause issues
        for copying molecules with groups)

        Interactions for ``Atom`` objects may be reordered with respect to
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
            elif k in ('_bonded_interaction_pairs', '_nonbonded_interactions'):
                pass
            elif k == '_structure_list':
                # Separate structures into atoms and composites
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
                        if isinstance(inter, BondedInteraction):
                            memo[id(inter)] = inter
                    new_atom = deepcopy(atom, memo)
                    struct_map[atom] = new_atom

                # Create interactions
                new_interactions = []
                for inter, pair in self.bonded_interaction_pairs:
                    # try/except accounts for interactions associated with atoms
                    # that are in a composite subunit
                    try:
                        new_pair = [struct_map[atom] for atom in pair]
                        # Instead of adding atoms, replace the interaction with new one in copy
                        bond_type = type(inter)
                        new_interactions.append(bond_type(tuple(new_pair), **vars(inter)))
                    except KeyError:
                        pass

                for composite in composites:
                    new_composite = deepcopy(composite, memo)
                    struct_map[composite] = new_composite

                # List comprehension ensures order of structures in new
                # structure is the same as in original
                setattr(unit, k, [struct_map[s] for s in self._structure_list])
                setattr (unit, "_bonded_interactions", new_interactions)
            else:
                setattr(unit, k, deepcopy(v, memo))
        return unit

    def __str__(self) -> str:
        """
        Returns
        -------
        str
            The formula and center of mass position of the
            ``CompositeStructure``
        """

        name = self.name + ' ' if self.name else ''
        return (f"{name}{self.__class__.__name__}  "
                f"formula: {self.formula}  position: {self.position}")

    @property
    @abstractmethod
    def nonbonded_interactions(self) -> list[Interaction]:

        raise NotImplementedError

    @property
    @abstractmethod
    def bonded_interaction_pairs(self) -> list[Interaction]:

        raise NotImplementedError

    @property
    def formula(self) -> str:
        """
        Get the chemical formula of the ``CompositeStructure``

        Returns
        -------
        str
            The chemical formula using the Hill system
        """

        return get_reduced_chemical_formula([atom.element for atom
                                             in self.atoms])

    @property
    def universe(self) -> Union[Universe, None]:
        """
        Get or set the ``Universe`` to which the ``CompositeStructure``
        belongs

        Returns
        -------
        Universe
            The Universe to which the ``CompositeStructure`` belongs or
            `None`
        """

        try:
            return self._universe()
        except TypeError:
            return self._universe

    @universe.setter
    def universe(self, value: Universe) -> None:

        try:
            self._universe = weakref.ref(value)
        except TypeError:
            self._universe = None

        # If top level structure then set the universe of all subunits
        if self.top_level_structure == self:
            for structure in self.structure_list:
                structure.universe = value

    @property
    def structure_list(self) -> 'list[Structure]':
        """
        Get or set the ``Structure`` objects that are subunits of this
        ``CompositeStructure``

        Returns
        -------
        list
            `list` of ``Structure`` that are subunits of this
            ``CompositeStructure``
        """

        return self._structure_list

    @structure_list.setter
    def structure_list(self, value: 'list[Structure]') -> None:

        self._structure_list = value

    def copy(self, position, rotation=None) -> CompositeStructure:
        # pylint:disable=arguments-differ
        # CompositeStructure's can be rotated, which is meaningless for
        # Structures in general
        """
        Copies the ``CompositeStructure`` and all attributes, except ``ID``
        which is generated

        Copying a ``CompositeStructure`` (e.g. a ``Molecule``) will copy
        all of the ``Atom`` objects within it. All of these atoms will have
        identical bonded and nonbonded interactions to the
        ``CompositeStructure`` from which they were copied i.e. the
        ``CompositeStructure`` will be exacltly duplicated. The only
        attributes of the ``CompositeStructure`` which will differ are the
        ``position`` (which is passed as a Parameter to ``copy``), and the
        ``ID``, which is generated automatically.

        This will not currently work if the ``CompositeStructure`` has any
        bonded interactions with atoms external to it (e.g. it may cause issues
        for copying molecules with groups)

        Interactions for ``Atom`` objects may be reordered with respect to
        initial atoms

        Parameters
        ----------
        position : list, tuple, numpy.ndarray
            3 element `list`, `tuple` or ``array`` of `float` specifying the
            ``position`` of the new ``Structure``
        rotation : list, tuple, numpy.ndarray, optional
            3 element `list`, `tuple` or ``array`` of `floats` specifying the
            degrees of anticlockwise rotation around the x, y, and z axes
            respectively. The rotation is centered on the center of mass of the
            ``CompositeStructure``. The default ``rotation`` is `None`,
            which applies no rotation to the copied ``CompositeStructure``.

        Returns
        -------
        CompositeStructure
            A ``CompositeStructure`` of the same type with all non-unique
            attributes copied and a new ``position``
        """

        composite = super().copy(position)
        if rotation is not None:
            composite.rotate(x=rotation[0], y=rotation[1], z=rotation[2])
        return composite

    def _set_subunit_positions(self) -> None:

        """
        Sets the position of all subunits in the global frame in units of
        ``Ang``
        """

        for atom in self.atoms:
            atom.position = self.position + self._CoM_frame_positions[atom]

    def calc_CoM(self) -> np.ndarray:
        """
        Returns
        -------
        numpy.ndarray
            Position of the center of mass of the ``CompositeStructure`` in
            units of ``Ang``
        """

        mass = 0.
        weighted_positions = np.zeros(3)
        for atom in self.atoms:
            mass += atom.mass
            weighted_positions += (atom.position * atom.mass)
        return weighted_positions / mass

    def _calc_subunit_position_in_CoM_frame(self) -> None:
        """
        Calculate the position of all subunits in the
        ``CompositeStructure`` CoM frame in units of ``Ang``
        """
        # pylint: disable=attribute-defined-outside-init
        # _CoM_frame_positions breaks when defined in init

        CoM = self.calc_CoM()
        self._CoM_frame_positions = {atom: (atom.position - CoM) for atom in self.atoms}

    def rotate(self, x: float = 0., y: float = 0., z: float = 0.) -> None:
        """
        Rotates the ``CompositeStructure`` around its center of mass

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
        for atom in self.atoms:
            atom.position = (CoM
                             + rotation.apply(self._CoM_frame_positions[atom]))


@repr_decorator('name', 'ID', 'element', 'position', 'velocity')
class Atom(Structure):
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
        ``Ang / fs``. Note that if set, the velocity of atoms in the MD engine will
        be scaled when creating a `Simulation` in order to ensure the
        temperature is accurate. Otherwise, if the velocities of all `Atom`
        objects in a `Simulation` are 0, then the velocities of the atoms in
        the MD engine will be randomly chosen in order to provide an accurate
        temperature. The default is ``(0., 0., 0.)``.
    charge : float
        The charge of the ``Atom`` in units of elementary charge (``e``). The
        default is `None`, meaning that a ``Coulomb`` interaction is not applied
        to the ``Atom``.
    **settings
        ``mass`` (`float`)
            The atomic mass in ``amu``. If not provided a lookup table will be
            used.
        ``atom_type`` (`int`)
            All atoms with the same atom_type will have the same non-bonded interactions
            applied to them. If not provided, MDMC will automatically infer/assign atom types.
            All atoms with the same element and interactions will have the same atom type.
        ``name`` (`str`)
            A name to uniquely identify the atom. Used by ForceFields (e.g. OPLSAA). Atoms
            should only have the same names if they are equivalent. Defaults to the element
            of the atom.
        ``cutoff`` (`float`)
            Sets the cutoff radius in ``Ang``, beyond which this atom does not interact
            with other atoms. Must be set if a charge is being added to the atom.
    Attributes
    ----------
    element : str
        The atomic element label
    """

    def __init__(self, element: str,
                 position: Union['list[float]', 'tuple[float]', np.ndarray]
                 = (0., 0., 0.),
                 velocity: Union['list[float]', 'tuple[float]', np.ndarray]
                 = (0., 0., 0.),
                 charge: float = None, **settings: dict):

        self.universe = None
        # the syntax for optional keyword arguments is: kwargs.get(str, default_value)
        super().__init__(position, velocity, name=settings.get('name', element))
        self._nonbonded_interactions = []
        self._bonded_interaction_pairs = []
        self.element = element

        try:
            self.mass = settings['mass']
        except KeyError:
            self.mass = atom_properties.MASS[self.element]

        self._atom_type = settings.get('atom_type', None)
        self.cutoff = settings.get('cutoff', None)
        self.charge = charge

    def __deepcopy__(self, memo: dict) -> Atom:
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

    def __repr__(self) -> tuple:
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

    def __str__(self) -> str:
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
    def atoms(self) -> 'list[Atom]':
        """
        Get a `list` of the atoms, just consisting of the ``Atom``

        Returns
        -------
        list
            A `list` with a single item, the ``Atom``
        """

        return [self]

    @property
    def universe(self) -> Union[Universe, None]:
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
    def universe(self, value: Universe) -> None:

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
    def charge(self) -> Union[float, None]:
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
                    try:
                        value = interaction.parameters['charge'].value
                    except KeyError as error:
                        raise ValueError('Coulombic interaction does not have a'
                                            ' parameter "charge".') from error
            return value
        except AttributeError:
            return None

    @charge.setter
    @unit_decorator(unit=units.CHARGE)
    def charge(self, value: float) -> None:

        for inter in self.interactions:
            if isinstance(inter, Coulombic):
                if value is not None:
                    try:
                        inter.parameters['charge'].value = value
                    except KeyError as error:
                        raise ValueError('Coulombic interaction does not have'
                                         ' a parameter "charge".') from error
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
            if self.cutoff is None:
                warnings.warn("No cutoff was set for the Coulombic interaction of this atom."
                              " The default cutoff of 10 Angstrom will be used. To set a cutoff,"
                              " provide the argument cutoff=[value]"
                              " when initialising the Atom object.")
                self.cutoff = 10.
            Coulombic(atoms=self, charge=value, cutoff=self.cutoff)

    @property
    def mass(self) -> float:
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
    def mass(self, mass: float) -> None:

        self._mass = mass

    @property
    def atom_type(self) -> int:
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
    def atom_type(self, value: int) -> None:

        if self._atom_type:
            raise AttributeError('Can\'t change atom_type once it has been set')
        self._atom_type = value

        # Update atom_types in Coulombic interactions
        for inter in self.nonbonded_interactions:
            if isinstance(inter, Coulombic) and value not in inter.atom_types:
                inter.atom_types.append(value)

    @property
    def nonbonded_interactions(self) -> list[NonBondedInteraction]:
        """
        Get a `list` of the nonbonded interactions acting on the ``Atom``

        Returns
        -------
        list
            ``NonBondedInteractions`` acting on the ``Atom``
        """

        return self._nonbonded_interactions

    @property
    def bonded_interaction_pairs(self) -> list:
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

    def copy(self, position: Union['list[float]', 'tuple[float]', np.ndarray]) -> Atom:
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

    def add_interaction(self, interaction: Interaction, from_interaction: bool = False) -> None:
        """
        Adds an interaction to the ``Atom``

        Parameters
        ----------
        interaction : MDMC.MD.interactions.Interaction
            Any class dervied from ``Interaction``, or any object with base
            class ``Interaction``.  If an ``Interaction`` class is passed then
            it must be a ``NonBondedInteraction`` i.e. only takes a single
            ``Atom`` as an argument. If an ``Interaction`` object is passed then
            this ``Atom`` must be in the ``interaction.atoms``.
        from_interaction : bool, optional
            Specifies if this method has been called from an ``Interaction``.
            Default is `False`.
        """

        if issubclass(type(interaction), BondedInteraction):
            # The tuple most recently added to interaction.atoms should always
            # contain self
            if from_interaction:
                if not interaction.atoms or self not in interaction.atoms[-1]:
                    raise ValueError('incorrect atom_tuple passed to atom')
            else:
                interaction.add_atoms(self, from_structure=True)
            pair = (interaction, interaction.atoms[-1])
            if pair not in self.bonded_interaction_pairs:
                self._bonded_interaction_pairs.append((interaction, interaction.atoms[-1]))
        else:
            if interaction not in self.nonbonded_interactions:
                self._nonbonded_interactions.append(interaction)

    def copy_interactions(self, atom: Atom, memo: dict = None) -> None:
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
        # pylint: disable=protected-access
        # because bonded_interaction_pairs is not designed to be set directly

        # default arguments are set at definition time (and not at call time)
        # so setting memo={} as the default value is dangerous, because the
        # dict will not be reset between calls of copy_interactions()
        if memo is None:
            memo = {}

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

    def is_equivalent(self, structure: Structure) -> bool:

        return isinstance(structure, type(self)) and \
               all([structure.element == self.element,
                    structure.mass == self.mass,
                    structure.charge == self.charge,
                    len(self.bonded_interactions) == len(structure.bonded_interactions),
                    len(self.nonbonded_interactions) == len(structure.nonbonded_interactions),
                    all(a.is_equivalent(b) for a, b in
                        zip(self.bonded_interactions, structure.bonded_interactions)),
                    all(a.is_equivalent(b) for a, b in
                        zip(self.nonbonded_interactions, structure.nonbonded_interactions))])


class Molecule(CompositeStructure):
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
        ``Ang / fs``. The default is ``(0., 0., 0.)``.
    name : str, optional
        The name of the structure. The default is `None`.
    **settings
        ``atoms`` (`list`)
            A `list` of ``Atom``  which will be included in the ``Molecule``
        ``interactions`` (`list`)
            A `list` of ``Interaction`` acting on atoms within the ``Molecule``.
            The ``interactions`` provides a convenience for declaring
            interactions on atoms when a ``Molecule`` is initialized. It is not
            required and is exactly equivalent to initializing the interactions
            prior to the ``Molecule``.
    """

    def __init__(self,
                 position: Union['list[float]', 'tuple[float]', np.ndarray] = None,
                 velocity: Union['list[float]', 'tuple[float]', np.ndarray] = (0, 0, 0),
                 name = None,
                 **settings: dict):

        self._structure_list = settings['atoms']
        for structure in self._structure_list:
            structure.parent = self
        self._calc_subunit_position_in_CoM_frame()
        if position is None:
            position = self.calc_CoM()
        super().__init__(position, velocity, name)

    @property
    def position(self) -> np.ndarray:
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
    def position(self, position: Union['list[float]', 'tuple[float]', np.ndarray]) -> None:

        self._position = position
        self._set_subunit_positions()

    @property
    def nonbonded_interactions(self) -> 'list[NonBondedInteraction]':
        """
        Get a list of the nonbonded interactions acting on the ``Molecule``

        Returns
        -------
        list
            ``NonBondedInteraction`` objects acting on the ``Molecule``
        """

        return [inter for atom in self.atoms
                for inter in atom.nonbonded_interactions]

    @property
    def bonded_interaction_pairs(self) -> list:
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
        # pylint: disable=simplifiable-condition
        # as simplifying the condition breaks it for some reason

        # Cache only most recent value, as atoms only expected to increase
        @lru_cache(maxsize=1)
        def get_bonded_interaction_pairs(atoms):
            # Preserve the order for consistent bonded_interaction_pairs
            used = set()
            return [pair for atom in atoms for pair
                    in atom.bonded_interaction_pairs if pair not in used
                    and (used.add(pair) or True)]

        # Cast to tuple required so that it is hashable for lru_cache
        return get_bonded_interaction_pairs(tuple(self.atoms))

    @property
    @unit_decorator_getter(unit=units.MASS)
    def mass(self) -> float:
        """
        Get the molecular mass of the ``Molecule`` in ``amu``

        Returns
        -------
        float
            The molecular mass in ``amu``
        """

        mass = 0
        for atom in self.atoms:
            mass += atom.mass

        return mass

    @property
    @unit_decorator_getter(unit=units.CHARGE)
    def charge(self) -> float:
        """
        Get the total charge of the ``Molecule`` in ``e``.

        Returns
        -------
        float
            The total charge in ``e``
        """

        return sum(atom.charge for atom in self.atoms if atom.charge is not None)

    def is_equivalent(self, structure: Structure) -> bool:

        return isinstance(structure, type(self)) and \
               all([structure.formula == self.formula,
                structure.mass == self.mass,
                structure.charge == self.charge,
                len(self.bonded_interactions) == len(structure.bonded_interactions),
                len(self.nonbonded_interactions) == len(structure.nonbonded_interactions),
                all(a.is_equivalent(b) for a, b in
                        zip(self.bonded_interactions, structure.bonded_interactions)),
                all(a.is_equivalent(b) for a, b in
                        zip(self.nonbonded_interactions, structure.nonbonded_interactions))])


@repr_decorator('min', 'max', 'volume')
class BoundingBox:

    """
    A box with the minimum and maximum extents of the positions of a collection
    of atoms

    Parameters
    ----------
    atoms : list
        ``Atom`` objects for which the minimum and maximum extents are
        determined
    """

    def __init__(self, atoms: 'list[Atom]'):
        if not atoms:
            raise ValueError("Empty atoms passed; "
                             "it must contain at least one atom to create a BoundingBox object.")

        # Start with arbitrary min and max from the positions of the atoms in
        # the atom list
        self.min = self.max = atoms[0].position
        for atom in atoms[1:]:
            self.min = np.minimum(self.min, atom.position)
            self.max = np.maximum(self.max, atom.position)

    @property
    def min(self) -> np.ndarray:
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
    def min(self, value: np.ndarray) -> None:

        self._min = value

    @property
    def max(self) -> np.ndarray:
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
    def max(self, value: np.ndarray) -> None:

        self._max = value

    @property
    @unit_decorator_getter(unit=units.LENGTH ** 3)
    def volume(self) -> float:
        """
        Get the volume of the bounding box, in units of ``Ang ^ 3``

        Returns
        -------
        float
            The volume of the bounding box
        """

        return abs(np.prod(self.max - self.min))


def filter_atoms(atoms: 'list[Atom]', predicate: Callable) -> 'list[Atom]':
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


def filter_atoms_element(atoms: 'list[Atom]', element: str) -> 'list[Atom]':
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


def get_reduced_chemical_formula(symbols: 'list[str]',
                                 factor: int = None,
                                 system: str = 'Hill') -> str:
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
