"""Module in which all structural units are defined.

Atoms are the fundamental structural unit in terms of which all others must be
defined.  All shared behaviour is included within the StructuralUnit base class.

AUTHOR :    Thomas Farmer        START DATE :    2018-4-26 12:11:03"""

from abc import ABC, abstractmethod
import numpy as np
from functools import reduce
from itertools import count

class StructuralUnit(ABC):
    """Abstract base class for all structural units

 	Attributes:
 	ID - a unique identifier for each structural unit
    type - type of structural unit
    position - Position of center of mass
    velocity - average velocity
    bonds - list of all
    """

    # TODO: Replace ID generation method
    _ID_generator = count(start=1,step=1)

    # TODO: If structures are copied, assign new ID to copy
    def __init__(self):
        # TODO: Add init docstring
        self.ID = next(self._ID_generator)
        self._atoms = []
        self._interactions = set()
        self.type = type(self)
        self.universe = None

    def atom_list(self):
        return self._atoms

    def atom_iterator(self):
        return iter(self._atoms)

    def position(self):
        pass

    def velocity(self):
        pass

    @abstractmethod
    def add_interaction(self):
        raise NotImplementedError

    def interaction_set(self):
        return self._interactions


class Atom(StructuralUnit):
    """A single atom

    DESCRIPTION

    Attributes:
    atomID - Unique positive integer
    position - position in simulation box
    velocity -
    element - atomic element
    mass - atomic mass (amu).  Can either be specified of determined from loopup
    """

    def __init__(self,element,position=(0,0,0),velocity=(0,0,0),**kwargs):
        """init with position, velocity, element, mass and a non-bonded
        interaction

        The non-bonded interaction will be set as a Coulomb interaction once a
        force field has been applied to the universe and each element has an
        associated charge.
        """
        # TODO: Create lookup table for atomic masses
        # TODO: Check position and velocity are valid

        super().__init__()
        self.element = element
        self.mass = kwargs['mass']
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.add_interaction(NonBonded)

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

    def atom_list(self):
        return [self]


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

    # TODO: Make Molecule init from list of atoms and also list of element symbols (both with list of bonds)
    def __init__(self,**kwargs):
        super().__init__()
        self._atoms = kwargs['atoms']
        self._interactions = kwargs['interactions']

    def add_interaction(self):
        pass

# TODO: Take out atom operations common to both structures and interactions, like atom_list, into a mixin class

class Interaction(object):
    """Base class for interactions, both bonded, non-bonded and constraints

    When atoms are passed to interactions, the interaction is also added to the
    atoms.

    Attributes:
    atoms
    interaction_function - A class of bond interaction function (e.g. harmonic
    potential)
    parameters - bond interaction parameters
    """

    def __init__(self,*atoms):
        # TODO: Iterate over atoms adding self to atoms.interactions, maybe also with interaction type
        # TODO: Iterate over atoms adding each element to self.elements
        # TODO: Test that number of parameters is what is required by bond_interaction
        # TODO: Change init so that Interaction can be called with a structural unit rather than atom
        # TODO: Change from passing atoms to something more general, as other interactions also need to be passed (e.g. bonds for dihedral)
        self._atoms = list(atoms)
        self.parameters = None
        self.interaction_function = None
        self._generate_ID()
        self.universe = None

    # TODO: Unify atom_list methods
    # TODO: Extract code so that atoms is not regenerated each time
    def atom_list(self):
        atoms = []
        for atom in self._atoms:
            atoms.extend(atom.atom_list())
        return atoms

    def atom_IDs(self):
        return [atom.ID for atom in self.atom_list()]

    # TODO: Make IDs unique prime numbers
    def _generate_ID(self):
        self.ID = reduce(lambda x,y: x*y, self.atom_IDs())

    def sorted_element_list(self):
        """Returns elements sorted alphabetically"""
        return sorted([atom.element for atom in self.atom_list()])

    # TODO: Ensure this doesn't get called when interactions are added with a call to self from an atom object
    def _add_interaction_atoms(self):
        for atom in self.atom_list():
            atom.add_interaction(self)

class NonBonded(Interaction):
    """A non-bonded interaction

    Requires only a single atom, and can only take non-bonded functions as
    interaction functions.
    """

    def __init__(self,atom):
        super().__init__(atom)


class Bond(Interaction):
    """A bond between any two atoms

    Requires exactly two atoms.

    Attributes:
    """

    def __init__(self,atom1,atom2):
        super().__init__(atom1,atom2)


class BondAngle(Interaction):
    """A bond angle between any two bonds

    Requires either three atoms (rotation around central atom) or four atoms
    (rotation around central bond - dihedral or torsional rotation)

    Attributes:
    ATTRIBUTES"""

    def __init__(self):
        pass


# TODO: Think about whether a class which contains exclusions to non-bonded interactions is required
#class Exclusions(CompositeParameter):
