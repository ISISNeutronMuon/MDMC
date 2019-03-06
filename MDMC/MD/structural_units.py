"""Module in which all structural units are defined.

Atoms are the fundamental structural unit in terms of which all others must be
defined.  All shared behaviour is included within the StructuralUnit base class.

AUTHOR :    Thomas Farmer        START DATE :    2018-4-26 12:11:03"""

from abc import ABCMeta, abstractproperty
from copy import deepcopy
from itertools import count
from types import MethodType
import weakref

import numpy as np

import MDMC.common.atom_properties as atom_properties
from MDMC.common.decorators import unit_decorator
from MDMC.common import units
from MDMC.MD.interaction_functions import Coulomb


class StructuralUnit:

    """Abstract base class for all structural units

 	Attributes:
 	ID - a unique identifier for each StructuralUnit
    universe - the universe to which the StructuralUnit belongs
    structure_type - type of StructuralUnit
    name - a string specifying the name of the structure
    position - position of center of mass in units of Ang
    velocity - average velocity in units of Ang fs^-1
    bonds - list of all bonds
    parent - the StructuralUnit to which this unit belongs
    atom_list - a list of atoms belonging to the StructuralUnit
    interactions - a list of interactions acting on the StructuralUnit
    bonded_interaction_pairs - a list of (interaction, atoms) tuples where atoms
    is a list of atoms to which the bonded interaction applies. At least one of
    these atoms belongs to the StructuralUnit
    """

    __metaclass__ = ABCMeta

    # ID exists to facilitate a 1 to 1 association with structural units within
    # MD engines.  It may not be required or may only be required for atoms.
    _ID_generator = count(start=1, step=1)

    def __init__(self, position, velocity, name):

        """
        Arguments:
        position - a tuple or NumPy array specifying the position in units of
        Ang
        velocity - a tuple of NumPy array specifying the velocity in units of
        Ang fs^-1
        name - a string specifying the name
        """

        self.ID = self._generate_ID()
        self.position = position
        self.velocity = velocity
        self.name = name
        self.parent = self

    @property
    def position(self):

        return self._position

    @position.setter
    @unit_decorator(unit=units.LENGTH)
    def position(self, position):

        """
        Provides a warning if the specified position is not within the
        StructuralUnit's universe
        """

        # if not self.valid_position(position):
        #     raise RuntimeWarning("Warning: Structural unit lies outside of the"
        #                          "universe bounds")
        self._position = position

    @property
    def velocity(self):

        return self._velocity

    @velocity.setter
    @unit_decorator(unit=units.LENGTH / units.TIME)
    def velocity(self, velocity):

        self._velocity = velocity

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

    @abstractproperty
    def universe(self):

        """
        Returns:
        The universe to which the atom belongs or None
        """

        raise NotImplementedError

    def translate(self, displacement):

        """
        Translate the structural unit by the specified displacement

        Arguments:
        Displacement - three element tuple or NumPy array
        """

        self.position = self.position + np.array(displacement)

    @property
    def interactions(self):

        """
        A list of the interactions acting on the StructuralUnit
        """

        return self.bonded_interactions + self.nonbonded_interactions

    @property
    def bonded_interactions(self):

        """
        A list of the bonded interactions acting on the StructuralUnit
        """

        return [pair[0] for pair in self.bonded_interaction_pairs]

    @abstractproperty
    def nonbonded_interactions(self):

        """
        A list of the nonbonded interactions acting on the StructuralUnit
        """

        raise NotImplementedError

    @abstractproperty
    def bonded_interaction_pairs(self):

        """
        A list of (interaction, atoms) pairs acting on the StructuralUnit,
        where atoms is a tuple of all atoms for that specific bonded interaction

        Example:
        For an O Atom with two bonds, one to H1 and one to H2:

        print(O.bonded_interaction_pairs)
        [(Bond, (H1, O)),
         (Bond, (H2, O))]
        """

        raise NotImplementedError

    @property
    def structure_type(self):

        """
        Returns:
        string specifying the name of the class
        """

        return self.__class__.__name__

    def _generate_ID(self):

        """
        Uses class attribute to generate a unique ID for each StructuralUnit

        Returns:
        unique integer
        """

        return next(self._ID_generator)

    def top_level_structure(self):

        """
        Returns:
        Highest level StructuralUnit of which it is a member
        """

        if issubclass(type(self.parent), StructuralUnit) \
        and self.parent is not self:
            return self.parent.top_level_structure()
        else:
            return self

    def _position_in_parent_CoM_frame(self):

        """
        Returns:
        Position in parent CoM frame with units of Ang, or None if it has no
        parent structure
        """

        if self.top_level_structure() is self:
            raise AttributeError("This structure has no parent")
        else:
            return self.position - self.parent._get_center_of_mass()

    def _added_to_structure(self):

        """
        Method is called if it becomes subunit of another StructuralUnit
        """

        self._position_in_parent = self._position_in_parent_CoM_frame()

    def valid_position(self, position=None):

        """
        Checks if the specified position is within the bounds of the
        StructuralUnit's universe, if it is associated with one

        Arguments:
        position - 3 element tuple or NumPy array with units of Ang or None.  If
        None then Atom's position is used.

        Returns:
        True if position is within universe or there is no associated universe.
        False if StructuralUnit has an associated universe but the position is
        not within its bounds.
        """

        if position is None:
            position = self.position
        try:
            # (0., 0., 0.) is defined as the origin for all universes
            if (np.any(position < np.array([0., 0., 0])) or
                    np.any(position > self.universe.dims)):
                return False
            elif np.any(position == np.float('nan')):
                raise ValueError('position of {0} is underdefined'.format(self))
            else:
                return True
        except AttributeError:
            # Not a member of a universe
            return True


class CompositeStructuralUnit(StructuralUnit):

    """
    Base class for structural units comprised of more than one atom

    Attributes:
    structure_list - a list of all of the StructuralUnits belonging to the
    CompositeStructuralUnit
    """

    __metaclass__ = ABCMeta

    def __init__(self, position, velocity, name):

        super(CompositeStructuralUnit, self).__init__(position, velocity, name)
        self.universe = None

    def __deepcopy__(self, memo):

        """
        Copies the CompositeStructuralUnit and all attributes, except ID which
        is generated

        This will not currently work if the CompositeStructuralUnit has any
        bonded interactions with atoms external to it (e.g. it may cause issues
        for copying molecules with groups)

        Interactions for Atoms may be reordered with respect to initial atoms

        Arguments:
        memo - the memo dict
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


    @property
    def universe(self):

        """
        Returns:
        The universe to which the atom belongs or None
        """

        try:
            return self._universe()
        except TypeError:
            return self._universe

    @universe.setter
    def universe(self, value):

        """
        Sets self.universe to a weakref to universe or None, and sets the
        universe for all subunits
        """

        try:
            self._universe = weakref.ref(value)
        except TypeError:
            self._universe = None

        # If top level structure then set the universe of all subunits
        if self.top_level_structure() == self:
            for structure in self.structure_list:
                structure.universe = value

    @property
    def structure_list(self):

        """
        A list of all StructuralUnits that are subunits of this
        CompositeStructuralUnit
        """

        return self._structure_list

    @structure_list.setter
    def structure_list(self, value):

        self._structure_list = value


class Atom(StructuralUnit):

    """
    A single atom

    Attributes:
    element - string specifying the atomic element label
    mass - float specifying the atomic mass (amu)
    charge - float specifying the charge (e) if one has been applied to the Atom
    """

    def __init__(self, element, position=(0., 0., 0.), velocity=(0., 0., 0.),
                 **settings):

        """
        init with position, velocity, element, mass and a non-bonded interaction

        The Coulombic interaction value (i.e. charge) is set when a force field
        is applied to the universe, in units of e.

        Arguments:
        element - string specifying the atomic element label
        position - a tuple or NumPy array specifying the position in units of
        Ang
        velocity - a tuple of NumPy array specifying the velocity in units of
        Ang fs^-1
        Settings:
        mass - float specifying the atomic mass in amu. If not provided a lookup
        table will be used.
        """

        self.universe = None
        super(Atom, self).__init__(position, velocity, name=element)
        self._nonbonded_interactions = []
        self._bonded_interaction_pairs = []
        self.element = element
        try:
            self.mass = settings['mass']
        except KeyError:
            self.mass = atom_properties.MASS[self.element]
        self._atom_type = settings.get('atom_type', None)

    def __deepcopy__(self, memo):

        """
        Copies the Atom and all attributes, except ID which is generated

        Interactions are copied but the copied atom is substituted for the
        original atom.  For BondedInteractions this means that the copied atom
        will be bonded to all atoms to which the original atom is bonded.

        Arguments:
        memo - the memo dict
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
        Returns the element, mass, charge, universe, position, velocity and
        names of all interactions that apply to this atom
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

        return '{0}'.format(self.element)

    @property
    def atom_list(self):

        """
        Returns:
        A list containing the atom
        """

        return [self]

    @property
    def universe(self):

        try:
            return self._universe()
        except TypeError:
            return self._universe

    @universe.setter
    def universe(self, value):

        try:
            self._universe = weakref.ref(value)

            # Update universe for all interactions if not previously set
            for inter in self.interactions:
                if not inter.universe:
                    inter.universe = value
        except TypeError:
            self._universe = None

    @property
    def charge(self):

        """
        Returns:
        If a force field has been defined then the charge parameter will have
        been set, and is returned in units of e. If no charge parameter exists
        then None is returned.
        """

        try:
            for interaction in self.interactions:
                if isinstance(interaction, Coulombic):
                    # Zero index parameter can be used as there should only be
                    # one parameter as each atom only has a single charge
                    return interaction.params[0].value
            return None
        except AttributeError:
            return None

    @charge.setter
    @unit_decorator(unit=units.CHARGE)
    def charge(self, value):

        charge_set = False
        for interaction in self.interactions:
            if isinstance(interaction, Coulombic):
                # Coulombic interactions only have a single parameter
                interaction.params[0].value = value
                charge_set = True
        if not charge_set:
            raise AttributeError('the atom must have a Coulombic interaction'
                                 ' with an InteractionFunction before the'
                                 ' charge can be set')

    @property
    def mass(self):

        return self._mass

    @mass.setter
    @unit_decorator(unit=units.MASS)
    def mass(self, mass):

        """
        Either assigns mass to self._mass or uses lookup table to determine
        mass if it is unspecified
        """

        self._mass = mass

    @property
    def atom_type(self):

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
        A list of NonBondedInteractions acting on the atom
        """

        return self._nonbonded_interactions

    @property
    def bonded_interaction_pairs(self):

        """
        A list of (interaction, atoms) pairs acting on the atom, where atoms is
        a tuple of all atoms for that specific interaction

        Example:
        For an O Atom with two bonds, one to H1 and one to H2:

        print(O.bonded_interaction_pairs)
        [(Bond, (H1, O)),
         (Bond, (H2, O))]
        """

        return self._bonded_interaction_pairs

    def add_interaction(self, interaction, from_interaction=False):

        """
        Adds an interaction to the atom

        Arguments:
        interaction - any class dervied from Interaction, or any object with
        base class Interaction.  If an interaction class is passed then it must
        be a non-bonded Interaction i.e. only takes a single atom as an
        argument. If an interaction object is passed then this atom must be in
        the interaction.atom_list.
        from_interaction - a boolean specifying if this method has been called
        from an interaction
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
        This replicates the interactions from self for atom, but with self
        substituted by atom in the atoms attribute for each interaction.  These
        interactions are added to any that already exist for the atom object.

        Passing the memo dictionary enables specific interactions to be excluded
        from being copied, duplicating the behaviour of __deepcopy__

        Arguments:
        atom - an atom object for which self.interactions are replicated
        memo - the memo dictionary
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


class Group(CompositeStructuralUnit):

    """
    Two or more atoms that form a subset of a molcule

 	Attributes:
 	position - center of mass position in units of Ang
    velocity - center of mass translational velocity in units of Ang fs^-1
    """

    def __init__(self):

        raise NotImplementedError


class Molecule(CompositeStructuralUnit):

    """
    Two or more bonded atoms, passed either as individual atoms or as groups

    Must be declared with at least 2 atoms.

    Attributes:
    position - center of mass position in units of Ang
    velocity - center of mass velocity in units of Ang fs^-1
    interactions - a set of interactions that involve any of the atoms within
    the Molecule
    bounding_box - a BoundingBox specifying the lower and upper extents of the
    Molecule
    """

    def __init__(self, position=(0, 0, 0), velocity=(0, 0, 0), name=None,
                 **settings):

        """
        Arguments:
        position - a tuple or NumPy array specifying the center of mass position
        in units of Ang
        velocity - a tuple of NumPy array specifying the center of mass velocity
        in units of Ang fs^-1
        name - a string specifying the name of the Molecule

        Settings:
        interactions - a list of interactions acting on atoms within the
        Molecule

        interactions keyword provides a convenience for declaring interactions
        on atoms when a Molecule is initialized. It is not required and is
        exactly equivalent to initializing the interactions prior to the
        Molecule.
        """

        self._structure_list = settings['atoms']
        for structure in self._structure_list:
            structure.parent = self
        self._calc_subunit_position_in_CoM_frame()
        super(Molecule, self).__init__(position, velocity, name)

    @property
    def position(self):

        return self._position

    @position.setter
    @unit_decorator(unit=units.LENGTH)
    def position(self, position):

        self._position = position
        self._set_subunit_positions()

    @property
    def nonbonded_interactions(self):

        """
        A list of NonBondedInteractions acting on atoms of the Molecule
        """

        return [inter for atom in self.atom_list
                for inter in atom.nonbonded_interactions]

    @property
    def bonded_interaction_pairs(self):

        """
        A list of (interaction, atoms) pairs acting on the StructuralUnit,
        where atoms is a tuple of all atoms for that specific interaction

        Example:
        For an O Atom with two bonds, one to H1 and one to H2:

        print(O.bonded_interaction_pairs)
        [(Bond, (H1, O)),
         (Bond, (H2, O))]
        """

        return list(set([pair for atom in self.atom_list
                         for pair in atom.bonded_interaction_pairs]))

    def _set_subunit_positions(self):

        """
        Sets the position of all subunits in the global frame in units of Ang
        """

        for atom in self.atom_list:
            atom.position = self.position + self._CoM_frame_positions[atom]

    def _calc_CoM(self):

        """
        Returns:
        position of the center of mass of the molecule in units of Ang
        """

        mass = 0.
        weighted_positions = np.zeros(3)
        for atom in self.atom_list:
            mass += atom.mass
            weighted_positions += (atom.position * atom.mass)
        return weighted_positions / mass

    def _calc_subunit_position_in_CoM_frame(self):

        """
        Calculate the position of all subunits in the Molecule CoM frame in
        units of Ang
        """

        self._CoM_frame_positions = {}
        CoM = self._calc_CoM()
        for atom in self.atom_list:
            self._CoM_frame_positions[atom] = atom.position - CoM

    @property
    def bounding_box(self):

        """
        Returns:
        BoundingBox specifying the lower and upper extents of the Molecule
        """

        return BoundingBox(self.atom_list)


class BoundingBox(object):

    """
    A box with the minimum and maximum extents of the atoms

    Attributes:
    min - a NumPy array with the minimum extent
    max - a NumPy array with the maximum extent
    """

    def __init__(self, atom_list):

        """
        Arguments:
        atom_list - a list of the Atoms which comprise the StructuralUnit in
        units of Ang
        """

        # Start with arbitrary min and max from the positions of the atoms in
        # the atom list
        self.min = self.max = atom_list.pop().position
        for atom in atom_list:
            self.min = np.minimum(self.min, atom.position)
            self.max = np.maximum(self.max, atom.position)

    @property
    def min(self):

        return self._min

    @min.setter
    @unit_decorator(unit=units.LENGTH)
    def min(self, value):

        self._min = value

    @property
    def max(self):

        return self._max

    @max.setter
    @unit_decorator(unit=units.LENGTH)
    def max(self, value):

        self._max = value


def filter_atoms(atoms, predicate):

    """
    Filters a list of atoms with a given predicate

    Arguments:
    parameters - a list of atoms
    predicate - a function that returns a boolean

    Returns:
    a list of atoms which meet the condition of the predicate
    """

    return filter(predicate, atoms)


def filter_atoms_element(atoms, element):

    """
    Filters a list of atoms based on the atomic element

    Arguments:
    atoms - a list of atoms
    element - a string specifying the atomic element label

    Returns:
    a list of atoms of a specific element
    """

    return filter(lambda a: a.element == element, atoms)


class Interaction:

    """
    Base class for interactions, both bonded, non-bonded and constraints

    Each different type of interaction should have an Interaction object. This
    object contains a list of all of the atoms (or atom pairs, triplets or
    quadruplets,  dependeing on the type of interaction) for which this
    Interaction applies. For example, an Ocoulombic interaction would contain a
    list of tuples where each tuple contains a different O atom, and a HObond
    interaction would contain a list of tuples where each tuple contains a
    different H and O pair. Interaction objects can be sliced to return a
    sublist of the tuples.

    When Atoms are passed to Interactions, the Interaction is also added to the
    Atoms.

    Attributes:
    atom_list - a list of the Atoms to which the Interaction is applied
    function - A class of InteractionFunction (e.g. HarmonicPotential)
    function_name - the name of the InteractionFunction
    universe - the Universe the interaction belongs to
    name - the name of the Interaction
    params - Interaction parameters
    """

    __metaclass__ = ABCMeta

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

        Settings:
        function - a class of bond interaction function (e.g. HarmonicPotential)
        """

        self.function = settings.get('function', None)
        self.name = self.__class__.__name__

    def __deepcopy__(self, memo={}):

        """
        Interactions cannot be copied

        Arguments:
        memo - the memo dict
        """

        raise AttributeError('Interactions cannot be copied')

    def __copy__(self):

        """
        Interactions cannot be copied
        """

        self.__deepcopy__()

    def __repr__(self):

        try:
            params = self.params
        except AttributeError:
            params = None

        return ('{0}'
                '  function: {1},'
                '  parameters: {2},'
                '  universe: {3},'
                '  elements: {4}'.format(self.name,
                                         self.function,
                                         params,
                                         self.universe,
                                         self.element_list()))

    def __str__(self):

        return self.__repr__

    @abstractproperty
    def atoms(self):

        raise NotImplementedError

    @property
    def params(self):

        try:
            return self.function.params
        except AttributeError:
            raise AttributeError('Interaction has no params as no force field'
                                 ' has been defined on it')

    @property
    def function(self):

        return self._function

    @function.setter
    def function(self, value):

        self._function = value

    @property
    def function_name(self):

        try:
            return self.function.name
        except AttributeError:
            return None

    @abstractproperty
    def universe(self):

        raise NotImplementedError

    @abstractproperty
    def element_list(self):

        """
        Returns:
        A list of elements for which the Interaction applies
        """

        raise NotImplementedError

    def sorted_element_list(self):

        """
        Returns:
        Elements sorted alphabetically, or None if the Interaction has not been
        applied to any atoms
        """

        return sorted(self.element_list())

    def element_tuple(self):

        """
        Returns:
        A tuple of element for which the Interaction applies, or None if the
        Interaction has not been applied to any atoms
        """

        return tuple(self.element_list())

    def _add_interaction_atoms(self, atoms):

        for atom in atoms:
            atom.add_interaction(self, from_interaction=True)


class NonBondedInteraction(Interaction):

    """
    Base class for non-bonded interactions

    Attributes:
    cutoff - a distance in Ang at which the interaction potential is truncated
    """

    __metaclass__ = ABCMeta

    def __init__(self, universe, *atom_types, **settings):

        self.universe = universe
        if self.universe:
            self.universe.add_nonbonded_interaction(self)
        self.cutoff = settings.get('cutoff')
        super(NonBondedInteraction, self).__init__(**settings)

    @abstractproperty
    def atom_types(self):

        raise NotImplementedError

    @property
    def universe(self):

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

        return self._cutoff

    @cutoff.setter
    @unit_decorator(unit=units.LENGTH)
    def cutoff(self, value):

        self._cutoff = value


class Dispersion(NonBondedInteraction):

    """
    A non-bonded dispersive interaction - either LJ or Buckingham
    """

    def __init__(self, universe, *atom_types, **settings):

        """
        Arguments:
        universe - a Universe object
        atom_types - one or two tuples containing one or more integers that

        Settings:
        vdw_tail_correction - a boolean specifying if the tail correction to the
        energy and pressure should be applied. This only affects the simulation
        dynamics if it is constant pressure.
        """

        super(Dispersion, self).__init__(universe, **settings)
        # Add tuples to short format of atom_types
        if isinstance(atom_types[0], int):
            if len(atom_types) == 1:
                atom_types = ((atom_types[0], ), (atom_types[0], ))
            elif len(atom_types) == 2:
                atom_types = ((atom_types[0], ), (atom_types[1], ))
        self._atom_types = atom_types
        self._atoms = [tuple([atom for atom_type in tpl
                              for atom in self.universe.atom_types[atom_type]])
                       for tpl in self.atom_types]
        for tpl in self.atoms:
            for atom in tpl:
                atom.add_interaction(self)

        self.vdw_tail_correction = settings.get('vdw_tail_correction', False)

    @property
    def atom_types(self):

        return self._atom_types

    @property
    def atoms(self):

        """
        Returns:
        A list of two tuples, where each tuple contains a list atoms. Every atom
        in the first tuple has a dispersion interaction with every atom in the
        second tuple (excluding self interactions). This is the complete list of
        possible dispersion interactions, i.e. it is only exactly correct if no
        cutoff has been specified.
        """

        return self._atoms

    def element_list(self):

        """
        Returns:
        A list of elements for which the Interaction applies
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
    """

    def __init__(self, universe=None, *atom_types, **settings):

        """
        Arguments:
        atom_types - one or more integers specifying atom_types that exist in
        the universe

        Settings:
        charge - a float specifying the charge parameter of the Coulombic
        interaction, in units of e. If this argument is passed, the inteaction
        function of this Coulombic object is set to a Coulomb interaction
        function with this float as its parameter. For example, the following
        initialization are equivalent:

        O = Atom('O', atom_type=1)
        O_coulombic = Coulombic(O.atom_type, charge=-0.84)
        O_coulombic = Coulombic(O.atom_type, function=Coulomb(-0.84))

        Passing a charge will overwrite any other interaction functions that are
        set, i.e. it makes the function keyword redundant
        atoms - one or more Atom objects
        """

        if atom_types:
            if not universe:
                raise TypeError('Coulombic requires a universe when atom_types'
                                ' are passed')
            super(Coulombic, self).__init__(universe, **settings)
            self.add_atom_types = MethodType(_add_atom_types, self)

            self._atom_types = atom_types
            self._atoms = [atom for atom_type in self.atom_types
                           for atom in self.universe.atom_types[atom_type]]
            # Add interaction to atoms
            for atom in self.atoms:
                atom.add_interaction(self)
        else:
            self.add_atoms = MethodType(_add_atoms, self)
            try:
                atoms = settings['atoms']
            except KeyError:
                raise TypeError('Coulombic takes either atom_types or atoms as'
                                ' arguments')
            # Account for init argument atoms=atom rather than atoms=[atom]
            if isinstance(atoms, Atom):
                atoms = [atoms]
            self._atoms = []
            self._atom_types = []
            self.add_atoms(*atoms)

            # Assumes all atoms are in the same universe (or None)
            universe = self.atoms[0].universe
            super(Coulombic, self).__init__(universe, **settings)

        charge = settings.get('charge')
        if charge:
            # Initializes a Coulomb interaction function with charge and units
            # and assigns it to self.function
            self.function = Coulomb(units.UnitFloat(charge, units.CHARGE))

    def __len__(self):

        """
        Returns:
        The number of interactions of this type that have been set
        """

        return len(self.atoms)

    def __getitem__(self, key):

        """
        Returns:
        The tuple of atoms at the specified index.  For a single index (as
        opposed to a slice) this is a group of atoms for which there is one
        instance of this interaction.
        """

        return self.atoms[key]

    @property
    def atoms(self):

        return self._atoms

    @property
    def atom_types(self):

        """
        Returns:
        All atom_types to which the Coulombic interaction applies

        If the interaction was initialized with the atoms argument, all
        atom_types of the atoms to which the Coulombic interaction was applied
        are returned; HOWEVER THE COULOMBIC INTERACTION IS NOT APPLIED TO ALL
        ATOMS OF THESE ATOM_TYPES, ONLY THE ATOMS IN self.atoms
        """

        return self._atom_types

    def element_list(self):

        """
        Returns:
        A list of elements for which the Interaction applies
        """

        return list(set(atom.element for atom in self._atoms))


def _add_atom_types(self, *atom_types):

    """
    Function for dynamically creating an add_atom_types method in Coulombic

    Arguments:
    atom_types - one or more integers specifying atom_types that exist in
    universe of the Coulombic interaction
    """

    self._atom_types.append(*atom_types)


def _add_atoms(self, *atoms):

    """
    Function for dynamically creating an add_atoms method in Coulombic

    Adds atoms to Coulombic object and adds Coulombic object to atoms
    nonbonded_interactions

    Arguments:
    atoms - one or more atoms
    """


    for atom in atoms:
        # Add atom to interaction
        self._atoms.append(atom)
        # Add interaction to atom
        atom.add_interaction(self, from_interaction=True)
        # Add atom_type to interaction.atom_types
        if atom.atom_type and atom.atom_type not in self.atom_types:
            self._atom_types.append(atom.atom_type)


class BondedInteraction(Interaction):

    """
    Base class for bonded interactions
    """

    __metaclass__ = ABCMeta

    def __init__(self, *atom_tuples, **settings):

        """
        Arguments:
        atom_tuples - Either a list of atoms involved in a single interaction,
        or a list of tuples, where each tuples contains all the Atoms involved
        in that interaction. For example:

        BondedInteraction(H1, O1, H2)

        should be used for a single BondedInteraction applied to H1, O1 and H2
        Atoms. Alternatively:

        BondedInteraction((H1, O1, H2), (H3, O2, H4))

        should be used for two BondedInteractions, one applied to H1, O1 and H2
        Atoms, and the other applied to H3, O2 and H4 Atoms.

        For three or more Atoms the order of the Atoms is important.  For
        example the above BondedInteractions could bother specify a H-O-H
        BondAngle, where the following would specify a H-H-O BondAngle:

        BondAngle(H1, H2, O)

        Settings:
        n_atoms - an integer specifying the number of atoms to which this
        interaction applies, for example 2 for a Bond.
        """

        if atom_tuples and isinstance(atom_tuples[0], Atom):
            atom_tuples = (atom_tuples, )
        if settings.get('n_atoms'):
            # This ensures that BondedInteractions can also be __init__ with 0
            # atoms
            for tpl in atom_tuples:
                self._validate_atoms(tpl, settings.get('n_atoms'))
        self.atoms = list(atom_tuples)
        super(BondedInteraction, self).__init__(**settings)

    def __len__(self):

        """
        Returns:
        The number of interactions of this type that have been set
        """

        return len(self.atoms)

    def __getitem__(self, key):

        """
        Returns:
        The tuple of atoms at the specified index.  For a single index (as
        opposed to a slice) this is a group of atoms for which there is one
        instance of this interaction.
        """

        return self.atoms[key]

    @property
    def atoms(self):

        return self._atoms

    @atoms.setter
    def atoms(self, atom_tuples):

        """
        Arugments:
        atom_tuples - a list of tuples containing one or more atoms.  Each tuple
        contains all of the atoms involved in one example of the interaction.
        For example a BondAngle interaction each tuple would contain 3 or 4
        atoms.
        """

        # Check for duplicate tuples in list
        self._check_duplicates(atom_tuples, 'Each tuple in the list of atom'
                                            ' tuples must be unique')
        # Check for duplicate atoms in each tuple
        try:
            for tpl in atom_tuples:
                self._check_duplicates(tpl, 'Each atom in an atom tuple must be'
                                            ' unique')
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

        try:
            return self.atoms[0][0].universe
        except IndexError:
            return None

    def element_list(self):

        """
        Returns:
        A list of elements for which the Interaction applies or None if the
        Interaction has not been applied to any atoms
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
        """

        if len(atoms) not in n_atoms:
            raise TypeError("This interaction only accepts {0} atoms".format(
                n_atoms))

    def add_atoms(self, *atoms, **settings):

        """
        Add atoms which are all involved in one example of this interaction

        Arguments:
        *atoms - one or more Atom objects

        Settings:
        from_structure - a boolean specifying if this method has been called
        from a structural unit
        """

        self._check_duplicates(atoms, 'Each atom in an atom tuple must be'
                                      ' unique')
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

    def _check_duplicates(self, struct, err_msg):

        if len(set(struct)) != len(struct):
            raise ValueError(err_msg)

    def _add_to_universe(self, universe, tpl):

        """
        Adds interaction and atom tuple to universe
        """

        universe.add_bonded_interaction_pairs((self, tpl))


class Constrainable(object):

    """
    A mixin class enabling classes inheriting from BondedInteraction to be
    constrained

    These constraints are then applied by a constraint algorithm (e.g. SHAKE),
    which is specified in the Universe which the BondedInteraction belongs to.

    Attributes:
    constrained - a boolean specifying whether the object is constrained
    """

    def __init__(self, *atom_tuples, **settings):

        self.constrained = settings.get('constrained', False)
        super(Constrainable, self).__init__(*atom_tuples, **settings)


class Bond(Constrainable, BondedInteraction):

    """
    A bond between any two atoms. Requires exactly two atoms in each atom_tuple.
    """

    def __init__(self, *atom_tuples, **settings):

        settings['n_atoms'] = (2, )
        super(Bond, self).__init__(*atom_tuples, **settings)


class BondAngle(Constrainable, BondedInteraction):

    """
    A bond angle between any two bonds

    Requires either three atoms (rotation around central atom) or four atoms
    (rotation around central bond - dihedral or torsional rotation) in each
    atom_tuple.
    """

    def __init__(self, *atom_tuples, **settings):

        settings['n_atoms'] = (3, 4)
        super(BondAngle, self).__init__(*atom_tuples, **settings)
