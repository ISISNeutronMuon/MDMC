"""Module in which all structural units are defined.

Atoms are the fundamental structural unit in terms of which all others must be
defined.  All shared behaviour is included within the StructuralUnit base class.

AUTHOR :    Thomas Farmer        START DATE :    2018-4-26 12:11:03"""

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from functools import reduce
from itertools import count
import weakref

import numpy as np

import MDMC.common.atom_properties as atom_properties


class StructuralUnit:

    """Abstract base class for all structural units

 	Attributes:
 	ID - a unique identifier for each structural unit
    universe - the universe to which the structural unit belongs
    structure_type - type of structural unit
    name - a string specifying the name of the structure
    position - position of center of mass
    velocity - average velocity
    bonds - list of all bonds
    parent - the structural unit to which this unit belongs
    """

    __metaclass__ = ABCMeta

    # ID exists to facilitate a 1 to 1 association with structural units within
    # MD engines.  It may not be required or may only be required for atoms.
    _ID_generator = count(start=1, step=1)

    def __init__(self, position, velocity, name):

        """
        Attributes:
        position - a tuple or NumPy array specifying the position
        velocity - a tuple of NumPy array specifying the velocity
        name - a string specifying the name
        """

        self.ID = self._generate_ID()
        self._interactions = set()
        self.universe = None
        self.position = position
        self.velocity = velocity
        self.name = name
        self.parent = self

    def __deepcopy__(self, memo):

        """
        Copies the StructuralUnit and all attributes, except ID which is
        generated

        Arguments:
        memo - the memo dict
        """

        cls = self.__class__
        structural_unit = cls.__new__(cls)
        memo[id(self)] = structural_unit
        for k, v in self.__dict__.items():
            if k == 'ID':
                setattr(structural_unit, k, self._generate_ID())
            else:
                setattr(structural_unit, k, deepcopy(v, memo))
        return structural_unit

    @property
    def position(self):

        return self._position

    @position.setter
    def position(self, position):

        """
        Provides a warning if the specified position is not within the
        structural units universe
        """

        if not self.valid_position(position):
            print "Warning: Structural unit lies outside of the universe bounds"
        self._position = np.array(position)

    @property
    def velocity(self):

        return self._velocity

    @velocity.setter
    def velocity(self, velocity):

        self._velocity = np.array(velocity)

    @property
    def atom_list(self):

        """
        Returns:
        A list of all of the atoms in the structure by recursively calling
        atom_list for all substructures.
        """

        atom_list = []
        for structure in self._structure_list:
            atom_list.extend(structure.atom_list)
        return atom_list

    @property
    def universe(self):

        """
        Returns:
        A weakref to universe or None
        """

        try:
            return self._universe()
        except TypeError:
            return self._universe

    @universe.setter
    def universe(self, universe):

        """
        Sets self.universe to a weakref to universe or None
        """

        try:
            self._universe = weakref.ref(universe)
        except TypeError:
            self._universe = None

    def translate(self, displacement):

        """
        Translate the structural unit by the specified displacement

        Arguments:
        Displacement - three element tuple or NumPy array
        """

        self.position = self.position + np.array(displacement)

    @abstractmethod
    def add_interaction(self):

        raise NotImplementedError

    @property
    def interactions(self):

        """
        A set of the interactions acting on the structural unit
        """

        return self._interactions

    @property
    def structure_type(self):

         return self.__class__.__name__

    def _generate_ID(self):

        """
        Uses class attribute to generate a unique ID for each structural unit
        """

        return next(self._ID_generator)

    def add_interaction(self, interaction):

        """
        Adds an interaction to the structural unit

        Arguments:
        interaction - any object with base class Interaction
        """

        self._interactions.add(interaction)

    def top_level_structure(self):

        """
        Returns:
        Highest level structural unit of which it is a member
        """

        if issubclass(type(self.parent),StructuralUnit) \
        and self.parent is not self:
            return self.parent.top_level_structure()
        else:
            return self

    def _position_in_parent_CoM_frame(self):

        """
        Returns:
        Position in parent CoM frame or None if it has no parent structure.
        """

        if self.top_level_structure() is self:
            raise AttributeError("This structure has no parent")
        else:
            return self.position - parent._get_center_of_mass()

    def _added_to_structure(self):

        """
        Method is called if it becomes subunit of another structural_unit
        """

        self._position_in_parent = self._position_in_parent_CoM_frame()

    def valid_position(self, position):

        """
        Checks if the specified position is within the bounds of the structural
        unit's universe, if it is associated with one

        Arguments:
        position - 3 element tuple or NumPy array

        Returns:
        True if position is within universe or there is no associated universe.
        False if structural unit has an associated universe but the position is
        not within its bounds.
        """

        try:
            # (0,0,0) is defined as the origin for all universes
            if np.any(position < np.array([0,0,0])) or \
                np.any(position > self.universe.dims):
                return False
            else:
                return True
        except AttributeError:
            # Not a member of a universe
            return True


class Atom(StructuralUnit):

    """
    A single atom

    Attributes:
    element - string specifying the atomic element label
    mass - float specifying the atomic mass (amu)
    """

    def __init__(self, element, position=(0,0,0), velocity=(0,0,0), **kwargs):

        """
        init with position, velocity, element, mass and a non-bonded interaction

        The Coulombic interaction value (i.e. charge) is set when a force field
        is applied to the universe.

        Arguments:
        element - string specifying the atomic element label
        Settings:
        mass - float specifying the atomic mass. If not provided a lookup table
        will be used.
        """

        super(Atom,self).__init__(position, velocity, name=element)
        self.element = element
        self.mass = kwargs.get('mass', None)
        if self.mass is None:
            try:
                self.mass = atom_properties.MASS[element]
            except KeyError:
                raise KeyError("The mass of that element is not tabulated")
        self.add_interaction(Coulombic(self))

    @property
    def atom_list(self):

        """
        Returns:
        A list containing the atom
        """

        return [self]

    @property
    def charge(self):

        """
        Returns:
        If a force field has been defined then the charge parameter will have
        been set, and is returned.  If no charge parameter exists then None is
        returned.
        """

        try:
            for interaction in self.interactions:
                if type(interaction) == Coulombic:
                    # Zero index parameter can be used as there should only be
                    # one parameter as each atom only has a single charge
                    return interaction.params[0].value
            else:
                return None
        except AttributeError:
            return None


class Group(StructuralUnit):

    """
    Two or more atoms that form a subset of a molcule

 	Attributes:
 	position - center of mass position
    velocity - center of mass translational velocity
    """

class Molecule(StructuralUnit):

    """
    Two or more bonded atoms, passed either as individual atoms or as groups

    Must be declared with at least 2 atoms and one interaction.

    Attributes:
    position - center of mass position
    velocity - center of mass translational velocity
    """

    # TODO: Make Molecule init from list of atoms/groups and also list of element symbols (both with list of bonds)
    # TODO: Create method for adding atoms/groups after init
    def __init__(self, position=(0,0,0), velocity=(0,0,0), name=None, **kwargs):

        self._structure_list = kwargs['atoms']
        for structure in self._structure_list:
            structure.parent = self
        self._calc_subunit_position_in_CoM_frame()
        super(Molecule,self).__init__(position, velocity, name)
        self.interactions = kwargs['interactions']

    @property
    def position(self):

        return self._position

    @position.setter
    def position(self, position):

        self._position = np.array(position)
        self._set_subunit_positions()

    @property
    def interactions(self):

        return self._interactions

    @interactions.setter
    def interactions(self, interactions):

        self._interactions = set(interactions)
        for atom in self.atom_list:
            for interaction in atom.interactions:
                self._interactions.add(interaction)

    def _set_subunit_positions(self):

        for atom in self.atom_list:
            atom.position = self.position + self._CoM_frame_positions[atom]

    # TODO: Improve implementation which determines atomic positions relative to molecular CoM
    def _calc_CoM(self):

        mass = 0.
        weighted_positions = np.zeros(3)
        for atom in self.atom_list:
            mass += atom.mass
            weighted_positions += (atom.position * atom.mass)
        return weighted_positions / mass

    def _calc_subunit_position_in_CoM_frame(self):

        self._CoM_frame_positions = {}
        CoM = self._calc_CoM()
        for atom in self.atom_list:
            self._CoM_frame_positions[atom] = atom.position - CoM

    # TODO: Not sure this method is necessary!
    @property
    def bounding_box(self):

        return BoundingBox(self.position, self.atom_list)


class BoundingBox(object):

    def __init__(self, position, atom_list):

        self.min = position
        self.max = position
        for atom in atom_list:
            self.min = np.minimum(self.min, atom.position)
            self.max = np.maximum(self.max, atom.position)


# TODO: Separate interactions into own module
class Interaction:

    """
    Base class for interactions, both bonded, non-bonded and constraints

    When atoms are passed to interactions, the interaction is also added to the
    atoms.

    Attributes:
    atoms
    function - A class of bond interaction function (e.g. harmonic
    potential)
    parameters - bond interaction parameters
    """

    __metaclass__ = ABCMeta

    def __init__(self, atom, *atoms):

        self._atom_list = [atom] + list(atoms)
        self.function = None
        self.function_name = None
        self._generate_ID()
        self.universe = None
        self._add_interaction_atoms()
        self.name = self.__class__.__name__

    def __deepcopy__(self, memo):

        """
        Prevents deepcopy of function (type InteractionFunction) as the function
        (and associated parameters) are shared for all Interactions which only
        differ on the specific atoms on which they act (not on the element
        types).
        """

        cls = self.__class__
        interaction = cls.__new__(cls)
        memo[id(self)] = interaction
        shallow_copy_attr = ['function','function_name']
        for k, v in self.__dict__.items():
            if k in shallow_copy_attr:
                setattr(interaction, k, getattr(self, k))
            else:
                setattr(interaction, k, deepcopy(v, memo))
        interaction.function.set_params_interactions(interaction)
        return interaction

    @property
    def atom_list(self):

        return self._atom_list

    @atom_list.setter
    def atom_list(self, structures):

        for structure in structures:
            self._atom_list.extend(structure.atom_list)

    @property
    def params(self):

        try:
            return self.function.params
        except AttributeError:
            raise AttributeError('Interaction has no params as no force field'
                                 ' has been defined on it')

    def atom_IDs(self):

        return [atom.ID for atom in self.atom_list]

    # TODO: Make IDs unique prime numbers
    def _generate_ID(self):

        self.ID = reduce(lambda x,y: x*y, self.atom_IDs())

    def element_list(self):

        return [atom.element for atom in self.atom_list]

    def sorted_element_list(self):

        """
        Returns:
        Elements sorted alphabetically
        """

        return sorted(self.element_list())

    # TODO: Currently defined so that force_field INTERACTION can be hashed - change this
    def _element_tuple(self):

        return tuple(self.element_list())

    # TODO: Ensure this doesn't get called when interactions are added with a call to self from an atom object
    def _add_interaction_atoms(self):

        for atom in self.atom_list:
            atom.add_interaction(self)


class Dispersion(Interaction):

    """
    A non-bonded dispersive interaction - either LJ or Buckingham

    Requires only a single atom, and can only take non-bonded functions as
    interaction functions.
    """

    def __init__(self, atom):

        super(Dispersion,self).__init__(atom)


class Coulombic(Interaction):

    """
    A non-bonded coulombic interaction - either normal or modified Coulomb

    Requires only a single atom, and can only take non-bonded functions as
    interaction functions.
    """

    def __init__(self, atom):

        super(Coulombic,self).__init__(atom)

class Bond(Interaction):

    """
    A bond between any two atoms. Requires exactly two atoms.
    """

    def __init__(self, atom1, atom2):

        super(Bond,self).__init__(atom1,atom2)


class BondAngle(Interaction):
    """
    A bond angle between any two bonds

    Requires either three atoms (rotation around central atom) or four atoms
    (rotation around central bond - dihedral or torsional rotation)
    """

    def __init__(self, atoms):

        self._validate_atoms(atoms)
        super(BondAngle,self).__init__(*atoms)

    def _validate_atoms(self, atoms):

        """
        Validates that the correct number of atoms have been passed to the
        interaction
        """

        n_atoms = len(atoms)
        if n_atoms == 3 or n_atoms == 4:
            pass
        else:
            raise ValueError("BondAngle only accepts three or four atoms")
