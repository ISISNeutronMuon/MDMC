"""Module in which all structural units are defined.

Atoms are the fundamental structural unit in terms of which all others must be
defined.  All shared behaviour is included within the StructuralUnit base class.

AUTHOR :    Thomas Farmer        START DATE :    2018-4-26 12:11:03"""

from abc import ABCMeta, abstractmethod
import numpy as np
from functools import reduce
from itertools import count
import weakref

class StructuralUnit:
    """Abstract base class for all structural units

 	Attributes:
 	ID - a unique identifier for each structural unit
    type - type of structural unit
    position - Position of center of mass
    velocity - average velocity
    bonds - list of all
    """

    __metaclass__ = ABCMeta

    # TODO: Replace ID generation method
    _ID_generator = count(start=1,step=1)

    # TODO: If structures are copied, assign new ID to copy
    def __init__(self,position,velocity,name):
        # TODO: Add init docstring
        # TODO: Ensure that deepcopy doesn't copy self.ID
        self.ID = next(self._ID_generator)
        self._interactions = set()
        self.type = type(self)
        self.universe = None
        self.position = position
        self.velocity = velocity
        self.name = name

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self,position):
        self._position = np.array(position)

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self,velocity):
        self._velocity = np.array(velocity)

    @property
    def atom_list(self):
        return self._atom_list

    def translate(self, displacement):

        """
        Translate the structural unit by the specified displacement

        Arguments:
        Displacement - three dimensional array
        """

        self.position = self.position + np.array(displacement)

    @abstractmethod
    def add_interaction(self):
        raise NotImplementedError

    def interaction_set(self):
        return self._interactions

    def interaction_list(self):
        return list(self.interaction_set())

    def top_level_structure(self):
        """
        Returns:
        Highest level structural unit of which it is a member
        """
        if issubclass(type(self.parent),StructuralUnit):
            return self.parent.top_level_structure()
        else:
            return self

    def _position_in_parent_CoM_frame(self):
        if top_level_structure() is self:
            # TODO: Raise error rather than returning None
            return None
        else:
            return self.position - parent._get_center_of_mass()

    def _added_to_structure(self):
        """Method is called if it becomes subunit of another structural_unit"""
        self._position_in_parent = _position_in_parent_CoM_frame()

    # TODO: Test position of all atoms is within universe


class Atom(StructuralUnit):
    """A single atom

    DESCRIPTION

    Attributes:
    atomID - Unique positive integer
    position - three dimensional array of the position in simulation box
    velocity - three dimensional array of the velocity
    element - atomic element
    mass - atomic mass (amu).  Can either be specified of determined from lookup
    """

    def __init__(self, element, position=(0,0,0), velocity=(0,0,0), **kwargs):
        """init with position, velocity, element, mass and a non-bonded
        interaction

        The non-bonded interaction will be set as a Coulomb interaction once a
        force field has been applied to the universe and each element has an
        associated charge.
        """
        # TODO: Create lookup table for atomic masses
        # TODO: Check position and velocity are valid

        super(Atom,self).__init__(position, velocity, name=element)
        self.element = element
        self.mass = kwargs['mass']
        self.add_interaction(Coulombic)
        self._atom_list = [self]

    # TODO: Think about naming of add_interaction and update_interactions
    def add_interaction(self,interaction_type,*atom):
        # TODO: Add interaction to universe if universe != None
        """Add an interaction to self.universe, passing self as first parameter
        """
        self.update_interactions(interaction_type(self,*atom))

    def update_interactions(self,interaction):
        self._interactions.add(interaction)

    def interaction_set(self):
        return self._interactions

    @property
    def charge(self):
        try:
            for interaction in self.interaction_list():
                if type(interaction) == Coulombic:
                    return interaction.function.params['charge']
            else:
                return None
        except AttributeError:
            return None


class Group(StructuralUnit):
	"""Two or more atoms that form a subset of a molcule

 	DESCRIPTION

 	Attributes:
 	position - center of mass position
    velocity - center of mass translational velocity
    """

class Molecule(StructuralUnit):
    """Two or more bonded atoms, passed either as individual atoms or as groups

    Must be declared with at least 2 atoms and one interaction.

    Attributes:
    position - center of mass position
    velocity - center of mass translational velocity
    """
    # TODO: Check bond lengths are consistent with atom positions
    # TODO: Make Molecule init from list of atoms and also list of element symbols (both with list of bonds)
    def __init__(self, position=(0,0,0), velocity=(0,0,0), name=None, **kwargs):
        self._atom_list = kwargs['atoms']
        self._calc_subunit_position_in_CoM_frame()
        super(Molecule,self).__init__(position, velocity, name)
        self._interactions = set(kwargs['interactions'])
        # TODO: ENSURE THAT INTERACTIONS FROM CONSTITUENT ATOMS ARE ADDED

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self,position):
        self._position = np.array(position)
        self._set_subunit_positions()

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

    @property
    def bounding_box(self):
        return BoundingBox(self.position, self.atom_list)

    # TODO: Unify add/update interaction methods
    def add_interaction(self,interaction):
        self._interactions.add(interaction)


class BoundingBox(object):

    def __init__(self, position, atom_list):
        self.min = position
        self.max = position
        for atom in atom_list:
            self.min = np.minimum(self.min,atom.position)
            self.max = np.maximum(self.max,atom.position)

# TODO: Take out atom operations common to both structures and interactions, like atom_list, into a mixin class

class Interaction(object):
    """Base class for interactions, both bonded, non-bonded and constraints

    When atoms are passed to interactions, the interaction is also added to the
    atoms.

    Attributes:
    atoms
    function - A class of bond interaction function (e.g. harmonic
    potential)
    parameters - bond interaction parameters
    """

    def __init__(self,atom,*atoms):
        # TODO: Iterate over atoms adding self to atoms.interactions, maybe also with interaction type
        # TODO: Iterate over atoms adding each element to self.elements
        # TODO: Test that number of parameters is what is required by bond_interaction
        # TODO: Change init so that Interaction can be called with a structural unit rather than atom
        # TODO: Change from passing atoms to something more general, as other interactions also need to be passed (e.g. bonds for dihedral)
        self._atom_list = [atom] + list(atoms)
        self.function = None
        self._generate_ID()
        self.universe = None
        self._add_interaction_atoms()

    # TODO: Unify atom_list methods
    # TODO: Extract code so that atoms is not regenerated each time
    @property
    def atom_list(self):
        return self._atom_list

    @atom_list.setter
    def atom_list(self,atoms):
        for atom in atoms:
            self._atom_list.extend(atom.atom_list)

    def atom_IDs(self):
        return [atom.ID for atom in self.atom_list]

    # TODO: Make IDs unique prime numbers
    def _generate_ID(self):
        self.ID = reduce(lambda x,y: x*y, self.atom_IDs())

    def element_list(self):
        return [atom.element for atom in self.atom_list]

    def sorted_element_list(self):
        """Returns elements sorted alphabetically"""
        return sorted(self.element_list())

    # TODO: Currently defined so that force_field INTERACTION can be hashed - change this
    def _element_tuple(self):
        return tuple(self.element_list())

    # TODO: Ensure this doesn't get called when interactions are added with a call to self from an atom object
    def _add_interaction_atoms(self):
        for atom in self.atom_list:
            atom.update_interactions(self)


class Dispersion(Interaction):
    """A non-bonded dispersive interaction - either LJ or Buckingham

    Requires only a single atom, and can only take non-bonded functions as
    interaction functions.
    """

    def __init__(self,atom):
        super(Dispersion,self).__init__(atom)

class Coulombic(Interaction):
    """A non-bonded coulombic interaction - either normal or modified Coulomb

    Requires only a single atom, and can only take non-bonded functions as
    interaction functions.
    """

    def __init__(self,atom):
        super(Coulombic,self).__init__(atom)

class Bond(Interaction):
    """A bond between any two atoms

    Requires exactly two atoms.

    Attributes:
    """

    def __init__(self,atom1,atom2):
        super(Bond,self).__init__(atom1,atom2)


class BondAngle(Interaction):
    """A bond angle between any two bonds

    Requires either three atoms (rotation around central atom) or four atoms
    (rotation around central bond - dihedral or torsional rotation)

    Attributes:
    ATTRIBUTES"""

    def __init__(self,**kwargs):
        # TODO: Improve ability to deal with both atoms and bonds
        try:
            atoms = kwargs['atoms']
        except KeyError:
            pass
            # Deal with bonds

        super(BondAngle,self).__init__(*atoms)


# TODO: Think about whether a class which contains exclusions to non-bonded interactions is required
#class Exclusions(CompositeParameter):
