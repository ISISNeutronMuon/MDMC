"""Module in which all structural units are defined.

Atoms are the fundamental structural unit in terms of which all others must be
defined.  All shared behaviour is included within the StructuralUnit base class.

AUTHOR :    Thomas Farmer        START DATE :    2018-4-26 12:11:03"""

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from functools import reduce
from inspect import isclass
from itertools import count
import weakref

import numpy as np

import MDMC.common.atom_properties as atom_properties
from MDMC.common.decorators import unit_decorator
from MDMC.common import units


class StructuralUnit:

    """Abstract base class for all structural units

 	Attributes:
 	ID - a unique identifier for each structural unit
    universe - the universe to which the structural unit belongs
    structure_type - type of structural unit
    name - a string specifying the name of the structure
    position - position of center of mass in units of Ang
    velocity - average velocity in units of Ang fs^-1
    bonds - list of all bonds
    parent - the structural unit to which this unit belongs
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
        self._interaction_pairs = []
        self.universe = None
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
        structural units universe
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

        # Set the universe of all subunits
        if not isinstance(self, Atom):
            for atom in self.atom_list:
                atom.universe = universe

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
        A list of the interactions acting on the structural unit
        """

        return [pair[0] for pair in self._interaction_pairs]

    @property
    def interaction_pairs(self):

        """
        A list of (interaction, atoms) pairs acting on the structural unit,
        where atoms is a tuple of all atoms for that specific interaction

        Example:
        For an O Atom with two bonds, one to H1 and one to H2:

        print(O.interaction_pairs)
        [(Bond, (H1, O)),
         (Bond, (H2, O))]
        """

        return self._interaction_pairs

    @property
    def structure_type(self):

         return self.__class__.__name__

    def _generate_ID(self):

        """
        Uses class attribute to generate a unique ID for each structural unit
        """

        return next(self._ID_generator)

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
        Position in parent CoM frame with units of Ang, or None if it has no
        parent structure
        """

        if self.top_level_structure() is self:
            raise AttributeError("This structure has no parent")
        else:
            return self.position - self.parent._get_center_of_mass()

    def _added_to_structure(self):

        """
        Method is called if it becomes subunit of another structural_unit
        """

        self._position_in_parent = self._position_in_parent_CoM_frame()

    def valid_position(self, position=None):

        """
        Checks if the specified position is within the bounds of the structural
        unit's universe, if it is associated with one

        Arguments:
        position - 3 element tuple or NumPy array with units of Ang or None.  If
        None then Atom's position is used.

        Returns:
        True if position is within universe or there is no associated universe.
        False if structural unit has an associated universe but the position is
        not within its bounds.
        """

        if position is None:
            position = self.position
        try:
            # (0,0,0) is defined as the origin for all universes
            if (np.any(position < np.array([0, 0, 0])) or
                    np.any(position > self.universe.dims)):
                return False
            elif np.any(position == np.float('nan')):
                raise ValueError('position of {0} is underdefined'.format(self))
            else:
                return True
        except AttributeError:
            # Not a member of a universe
            return True

    def add_interaction(self, interaction):

        """
        Adds an interaction to the Molecule, if it has not previously been added
        for this combination of atoms

        Arguments:
        interaction - any object of any class derived from Interaction
        """

        pair = (interaction, interaction.atoms[-1])
        if pair not in self.interaction_pairs:
            self._interaction_pairs.append((interaction, interaction.atoms[-1]))


class CompositeStructuralUnit(StructuralUnit):

    """
    Base class for structural units comprised of more than one atom
    """

    __metaclass__ = ABCMeta

    def __init__(self, position, velocity, name):

        super(CompositeStructuralUnit, self).__init__(position, velocity, name)

    def __deepcopy__(self, memo):

        """
        Copies the StructuralUnit and all attributes, except ID which is
        generated

        This will not currently work if the composite strucural unit has any
        bonded interactions with atoms external to it (e.g. it may cause issues
        for copying molecules with groups)


        Arguments:
        memo - the memo dict
        """

        cls = self.__class__
        unit = cls.__new__(cls)
        memo[id(self)] = unit
        for k, v in self.__dict__.items():
            if k == 'ID':
                setattr(unit, k, self._generate_ID())
            if k == '_interaction_pairs':
                pass
            if k == '_structure_list':
                # Seperate structures into atoms and composites
                atoms, composites = [], []
                for s in self._structure_list:
                    (atoms if isinstance(s, Atom) else composites).append(s)

                # Create dict to map from current to new structures. This is
                # used both for creating interactions with correct new atoms,
                # and preserving the structures ordering in unit._structure_list
                struct_map = {}
                for atom in atoms:
                    # Add atom's interactions to memo so that these are not
                    # copied
                    for inter in atom.interactions:
                        memo[id(inter)] = inter
                    new_atom = deepcopy(atom, memo)
                    struct_map[atom] = new_atom

                # Create interactions
                for inter, pair in self.interaction_pairs:
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

                # List comprehension orders structures
                setattr(unit, k, [struct_map[s] for s in self._structure_list])
            else:
                setattr(unit, k, deepcopy(v, memo))
        return unit

    @property
    def structure_list(self):

        """
        A list of all structural units that are subunits of this composite
        structural unit
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
    """

    def __init__(self, element, position=(0, 0, 0), velocity=(0, 0, 0),
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

        super(Atom, self).__init__(position, velocity, name=element)
        self.element = element
        self.mass = settings.get('mass', atom_properties.MASS[self.element])
        self.add_interaction(Coulombic)

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
        been set, and is returned in units of e. If no charge parameter exists
        then None is returned.
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

        # The tuple most recently added to interaction.atoms should always
        # contain self
        assert self in interaction.atoms[-1], ('incorrect atom_tuple passed to'
                                               ' atom')
        if not from_interaction:
            interaction.add_atoms((self, ), from_structure=True)
        super(Atom, self).add_interaction(interaction)

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

        # if/else required for deepcopy (where _interaction_pairs attribute
        # doesn't exist). try/except not valid due to order of operations in
        # add_atoms method.
        if not hasattr(atom, '_interaction_pairs'):
            atom._interaction_pairs = []
        for inter, atoms in self.interaction_pairs:
            if id(inter) not in memo:
                # Maintains order of atoms except with substitution
                new_atoms = [atom if a == self else a for a in atoms]

                # Use add_atoms method to update attribute rather than setattr.
                # This ensures interaction is added to all other atoms as well
                inter.add_atoms(*new_atoms)
                memo[id(inter)] = inter

class Group(StructuralUnit):

    """
    Two or more atoms that form a subset of a molcule

 	Attributes:
 	position - center of mass position in units of Ang
    velocity - center of mass translational velocity in units of Ang fs^-1
    """

    def __init__(self):

        raise NotImplementedError


class Molecule(StructuralUnit):

    """
    Two or more bonded atoms, passed either as individual atoms or as groups

    Must be declared with at least 2 atoms and one interaction.

    Attributes:
    position - center of mass position in units of Ang
    velocity - center of mass velocity in units of Ang fs^-1
    interactions - a set of interactions that involve any of the atoms within
    the Molecule
    """

    def __init__(self, position=(0,0,0), velocity=(0,0,0), name=None,
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
        self.interactions = settings['interactions']
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
    def interactions(self):

        return [pair[0] for pair in self.interaction_pairs]

    @property
    def interaction_pairs(self):

        return list(set([pair for atom in self.atom_list
                         for pair in atom.interaction_pairs]))

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
        atom_list - a list of the atoms which comprise the structural unit in
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

    Each different interaction should have an Interaction object. This object
    contains a list of all of the atoms (or atom pairs, triplets or quadruplets,
    dependeing on the type of interaction) for which this Interaction applies.
    For example, an Ocoulombic interaction would contain a list of tuples where
    each tuple contains a different O atom, and a HObond interaction would
    contain a list of tuples where each tuple contains a different H and O pair.
    Interaction objects can be sliced to return a sublist of the tuples.

    When atoms are passed to interactions, the interaction is also added to the
    atoms.

    Attributes:
    atom_list - a list of the atoms which possess the interaction
    function - A class of bond interaction function (e.g. harmonic
    potential)
    function_name - the name of the interaction function
    universe - the universe the interaction belongs to
    name - the name of the interaction
    params - bond interaction parameters
    """

    __metaclass__ = ABCMeta

    def __init__(self, *atom_tuples, **settings):

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

        self.atoms = list(atom_tuples)
        self.function = settings.get('function', None)
        self.universe = None
        self.name = self.__class__.__name__

    def __deepcopy__(self, memo):

        """
        Prevents deepcopy of function (type InteractionFunction) as the function
        (and associated parameters) are shared for all Interactions which only
        differ on the specific atoms on which they act (not on the element
        types).

        Arguments:
        memo - the memo dict
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

        # Set interactions for parameters if interaction.function has been
        # defined
        try:
            interaction.function.set_params_interactions(interaction)
        except AttributeError:
            pass
        return interaction

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

    # Both of these need to be modified so that the atoms add the interaction
    # def __setitem__(self, key, value):
    #
    #     self._atoms[key] = value
    #
    # def __delitem__(self, key):
    #
    #     del self._atoms[key]

    @property
    def atoms(self):

        return self._atoms

    @atoms.setter
    def atoms(self, atom_tuples):

        """
        Arugments:
        atom_tuples - a list of tuples containing one or more atoms.  Each tuple
        contains all of the atoms involved in one example of the interaction.
        For example for a non-bonded interaction each tuple would contain a
        single atom, and for a BondAngle interaction each tuple would contain 3
        or 4 atoms.
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
            # to ._interaction_pairs for every atom in the tuple
            self._atoms.append(tpl)
            self._add_interaction_atoms(tpl)

    @property
    def params(self):

        try:
            return self.function.params
        except AttributeError:
            raise AttributeError('Interaction has no params as no force field'
                                 ' has been defined on it')

    @property
    def function_name(self):

        try:
            return self.function.name
        except AttributeError:
            return None

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

    def element_list(self):

        """
        Returns:
        A list of elements for which the Interaction applies or None if the
        Interaction has not been applied to any atoms
        """

        try:
            # Each tuple should contain the same elements, so first tuple's used
            return [atom.element for atom in self.atoms[0]]
        except AttributeError:
            return None

    def sorted_element_list(self):

        """
        Returns:
        Elements sorted alphabetically
        """

        return sorted(self.element_list())

    def element_tuple(self):

        return tuple(self.element_list())

    def _add_interaction_atoms(self, atoms):

        for atom in atoms:
            atom.add_interaction(self, from_interaction=True)

    def _check_duplicates(self, struct, err_msg):
        if len(set(struct)) != len(struct):
            raise ValueError(err_msg)


class NonBondedInteraction(Interaction):

    """
    Base class for non-bonded interactions
    """

    __metaclass__ = ABCMeta

    def __init__(self, *atom_tuples, **settings):

        if isinstance(atom_tuples[0], Atom):
            atom_tuples = tuple((atom, ) for atom in atom_tuples)
        super(NonBondedInteraction, self).__init__(*atom_tuples, **settings)


class Dispersion(NonBondedInteraction):

    """
    A non-bonded dispersive interaction - either LJ or Buckingham

    Requires only a single atom, and can only take non-bonded functions as
    interaction functions.
    """

    def __init__(self, *atom_tuples, **settings):

        """
        Arguments:
        atom_tuples - at Atom object
        """

        super(Dispersion, self).__init__(*atom_tuples, **settings)


class Coulombic(NonBondedInteraction):

    """
    A non-bonded coulombic interaction - either normal or modified Coulomb

    Requires only a single atom, and can only take non-bonded functions as
    interaction functions.
    """

    def __init__(self, *atom_tuples, **settings):

        """
        Arguments:
        atom - at Atom object
        """

        super(Coulombic, self).__init__(*atom_tuples, **settings)


class BondedInteraction(Interaction):

    """
    Base class for bonded interactions
    """

    __metaclass__ = ABCMeta

    def __init__(self, *atom_tuples, **settings):

        """
        Settings:
        n_atoms - an integer specifying the number of atoms to which this
        interaction applies, for example 2 for a Bond.
        """

        if isinstance(atom_tuples[0], Atom):
            atom_tuples = (atom_tuples, )
        if settings.get('n_atoms'):
            # This ensures that BondedInteractions can also be __init__ with 0
            # atoms
            for tpl in atom_tuples:
                self._validate_atoms(tpl, settings.get('n_atoms'))
        super(BondedInteraction, self).__init__(*atom_tuples, **settings)

    def _validate_atoms(self, atoms, n_atoms):

        """
        Validates that the correct number of atoms have been passed to the
        interaction
        """

        if len(atoms) not in n_atoms:
            raise TypeError("This interaction only accepts {0} atoms".format(
                n_atoms))


class Bond(BondedInteraction):

    """
    A bond between any two atoms. Requires exactly two atoms.
    """

    def __init__(self, *atom_tuples, **settings):

        """
        Arguments:
        atom1, atom2 - at Atom object
        """

        settings['n_atoms'] = (2, )
        super(Bond, self).__init__(*atom_tuples, **settings)


class BondAngle(BondedInteraction):

    """
    A bond angle between any two bonds

    Requires either three atoms (rotation around central atom) or four atoms
    (rotation around central bond - dihedral or torsional rotation)
    """

    def __init__(self, *atom_tuples, **settings):

        """
        Arguments:
        atom1, atom2, atom3 - an Atom object
        *atom4 - one or more Atom objects
        """

        settings['n_atoms'] = (3, 4)
        super(BondAngle, self).__init__(*atom_tuples, **settings)
