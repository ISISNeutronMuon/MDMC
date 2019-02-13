"""Module for setting up and running the simulation

 Classes for the simulation box, minimizer and integrator.

 AUTHOR :    Thomas Farmer        START DATE :    2018-4-30 13:01:04"""

from abc import ABCMeta, abstractproperty
from collections import defaultdict
from copy import deepcopy
from itertools import product, ifilterfalse, count

from enum import Enum
import numpy as np

from MDMC.common.decorators import unit_decorator, unit_decorator_getter
from MDMC.common import units
from MDMC.MD.engine_facades.facade_factory import MDEngineFacadeFactory
from MDMC.MD.force_fields.force_field_factory import ForceFieldFactory
from MDMC.trajectory_analysis.trajectory import Configuration


Shape = Enum('Shape', ['cubic', 'orthorhombic', 'infinite',
                       'rhombic_dodecahedron', 'truncated_octahedron'])


class Universe(object):

    """
    Class where configuration and topology are defined

    Attributes:
    shape - member of the Shape enum
    dims - array of dimensions
    interactions - a list of interactions which exist in the universe
    bonded_interaction_pairs - a list of (interaction, atoms) tuples where atoms
    is a list of atoms to which the bonded interaction applies
    parameters - a list of interaction potential parameters
    volume - The volume of the universe
    element_list - A list of the elements in the universe
    element_dict - A dictionary of element:atom, where atom is a single atom of
    the specified element
    atom_list - a list of the atoms in the universe
    molecule_list - a list of the molecules in the universe
    structure_list - a list of the structural units in the universe
    force_fields - a list of the force fields that apply to the universe
    kspace_solver - a KSpaceSolver object specifying the k-space solver to
    be used for both electrostatic and dispersive interactions
    electrostatic_solver - a KSpaceSolver object specifying the k-space
    solver to be used for electrostatic interactions
    dispersive_solver - a KSpaceSolver object specifying the k-space solver
    to be used for dispersive interactions
    constraint_algorithm - an object which has a  ConstraintAlgorithm base
    class which specifies the constraint algorithm which will be applied to
    constrained BondedInteractions.
    """

    def __init__(self, dimensions, shape=Shape.cubic, force_field=None,
                 structures=None, **settings):

        """
        Arguments:
        dimensions - single float for cubic universe or 3 element list of
        floats, with units of Ang.
        shape - member of shape enum
        force_field - a subclass of MDMC.MD.force_fields.ff.ForceField
        structures - a list of structures

        Settings:
        kspace_solver - a KSpaceSolver object specifying the k-space solver to
        be used for both electrostatic and dispersive interactions. If this is
        passed then no electrostatic_solver or dispersive_solver may be
        provided.
        electrostatic_solver - a KSpaceSolver object specifying the k-space
        solver to be used for electrostatic interactions
        dispersive_solver - a KSpaceSolver object specifying the k-space solver
        to be used for dispersive interactions
        constraint_algorithm - an object which has a  ConstraintAlgorithm base
        class which specifies the constraint algorithm which will be applied to
        constrained BondedInteractions.
        """

        self.shape = shape
        self.dims = dimensions
        self._atom_types = defaultdict(list)
        self._atom_type_interactions = {}
        if structures:
            self.configuration = Configuration(structures)
        else:
            self.configuration = Configuration(universe=self)
        self._bonded_interaction_pairs = set()
        self._nonbonded_interactions = set()
        self.force_fields = force_field

        self.kspace_solver = settings.get('kspace_solver')
        self.electrostatic_solver = settings.get('electrostatic_solver')
        self.dispersive_solver = settings.get('dispersive_solver')
        # kspace_solver is mutually excusive with the other two solver
        # attributes
        if self.kspace_solver and (self.electrostatic_solver or
                                   self.dispersive_solver):
            raise ValueError('No other solver may be passed if kspace_solver is'
                             ' passed')

        self.constraint_algorithm = settings.get('constraint_algorithm')


    @property
    def dims(self):

        return self._dims

    @dims.setter
    @unit_decorator(unit=units.LENGTH)
    def dims(self, dims):

        if isinstance(dims, float):
            if self.shape == Shape.cubic:
                self._dims = np.array([dims] * 3)
            else:
                raise TypeError("Only dimensions of cubic Universes can be"
                                " specified with a float")
        elif isinstance(dims, (list, tuple, np.ndarray)):
            if len(dims) == 3:
                self._dims = np.array(dims)
            else:
                raise ValueError("3 dimensions must be specified")
        else:
            raise TypeError("dims must be a float or 3 element list of floats")

    @property
    def interactions(self):

        """
        A list of interactions in the universe
        """

        return self.bonded_interactions + self.nonbonded_interactions

    @property
    def bonded_interactions(self):

        """
        A list of the bonded interactions in the universe
        """

        return [pair[0] for pair in self.bonded_interaction_pairs]

    @property
    def nonbonded_interactions(self):

        """
        A list of the nonbonded interactions in the universe
        """

        return list(self._nonbonded_interactions)

    @property
    def bonded_interaction_pairs(self):

        """
        A list of (interaction, atoms) pairs in the universe, where atoms is a
        tuple of all atoms for that specific interaction

        Example:
        For an O Atom with two bonds, one to H1 and one to H2:

        print(O.bonded_interaction_pairs)
        [(Bond, (H1, O)),
         (Bond, (H2, O))]
        """

        # bonded_interaction_pairs is a set to avoid double counting of
        # interactions
        return list(self._bonded_interaction_pairs)

    @property
    def parameters(self):

        return set([param for interaction in self.interactions
                    for param in interaction.params])

    @property
    @unit_decorator_getter(unit=units.LENGTH ** 3)
    def volume(self):

        return np.prod(self.dims)

    @property
    def element_list(self):

        return [atom.element for atom in self.atom_list]

    @property
    def element_dict(self):

        """
        Returns a dictionary of all elements and a single atom of that
        element type. This is required for MD engines which assign the same
        potential parameters for all identical element types.
        """

        return {atom.element:atom for atom in self.atom_list}

    @property
    def atom_list(self):

        return self.configuration.atom_list

    @property
    def molecule_list(self):

        return self.configuration.molecule_list

    @property
    def structure_list(self):

        """
        Returns all structural units that exist in the Universe.  This includes
        all structural units that are a subunit of another structure belonging
        to the universe.
        """

        def add_all_parents(unit):

            parent = unit.parent
            parents = [parent]
            if parent is not parent.top_level_structure():
                parents += add_all_parents(parent)
            return parents

        structural_units = []
        for atom in self.atom_list:
            structural_units += add_all_parents(atom)

        structural_units += list(self.atom_list)
        return list(set(structural_units))

    @property
    def force_fields(self):

        return self._force_fields

    @force_fields.setter
    def force_fields(self, force_field):

        if force_field:
            self._force_fields = ForceFieldFactory.create_force_field(
                force_field)
        else:
            self._force_fields = None

    @property
    def atom_types(self):

        return self._atom_types

    @property
    def atom_type_interactions(self):

        return self._atom_type_interactions

    def _update_atom_types(self, atom):

        """
        Adds the atom to atom_types dictionary

        Arguments:
        atom - an Atom object to add to the atom_types dictionary
        """

        inter_key = (atom.element, ) + tuple(sorted(atom.interactions))
        if atom.atom_type:
            atom_type = atom.atom_type
            if atom_type not in self.atom_types:
                self._update_atom_type_interactions(inter_key, atom_type)
        else:
            try:
                atom_type = self.atom_type_interactions[inter_key]
            except KeyError:
                # Get lowest missing interger in self.atom_type_interactions
                atom_type = next(ifilterfalse(set(
                    self.atom_type_interactions.values()).__contains__,
                                              count(1)))
                self._update_atom_type_interactions(inter_key, atom_type)
            atom.atom_type = atom_type
        self._atom_types[atom_type].append(atom)


    def _update_atom_type_interactions(self, key, atom_type):

        """
        Adds a new key:atom_type to atom_type_interactions, if the key does not
        already exist

        Arguments:
        key - a tuple of (element, *interactions), where element is a string
        specifying the atomic element, and *interactions is one or more
        Interaction objects
        atom_type - an integer specifying the atom type
        """

        if key not in self.atom_type_interactions:
            self.atom_type_interactions[key] = atom_type
        else:
            raise TypeError('assignments cannot be made to'
                            ' atom_type_interactions keys which already possess'
                            ' values')

    def add_structural_unit(self, structural_unit, force_field=None):

        """
        Adds a single structural unit to the universe, with optional force field
        applying only to that structural unit
        """

        structural_unit.universe = self
        self.configuration.add_structural_unit(structural_unit)
        for atom in structural_unit.atom_list:
            self.add_bonded_interaction_pairs(*atom.bonded_interaction_pairs)
            self.add_nonbonded_interaction(*atom.nonbonded_interactions)
            self._update_atom_types(atom)

        if force_field:
            self.add_force_field(force_field, structural_unit.interactions)

    def fill(self, structural_unit, force_field=None, **settings):

        """
        A liquid-like filling of the universe independent of existing atoms

        Adds copies of structural_unit to existing configuration until universe
        is full.  As exclusion region is defined by the size of a bounding
        sphere, this method is most suitable for atoms or molecules with
        approximately equal dimensions.

        CURRENT APPROACH RESULTS IN NUMBER DENSITY DIFFERENT TO WHAT IS
        SPECIFIED DEPENDING ON HOW CLOSE CUBE ROOT OF N_MOLECULES IS TO AN INT.

        Arguments:
        structural_unit - any object with base class StructuralUnit
        force_field - Simultaneously applies a forcefield (base class
        ForceField)
        Settings:
        num_density - non-negative float specifying number density
        """

        n_units_xyz = self.dims / (1. / settings.get('num_density')) ** (1 / 3.)
        n_units_xyz = n_units_xyz.astype(int)

        positions = []
        # Determine the upper and lower bounds for structural unit with its
        # position (CoM) and its bounding box
        bounds = structural_unit.bounding_box
        mn = np.array((0., 0., 0.))
        mx = self.dims
        for i in range(len(self.dims)):
            positions.append(np.linspace(mn[i], mx[i], n_units_xyz[i],
                                         endpoint=False))

        positions = sorted(list(product(*positions)))

        # Add the first structural unit and force field (if specified) before
        # copying the structural unit to fill the universe
        for position in positions:
            if position is positions[0]:
                self.add_structural_unit(structural_unit, force_field)
                structural_unit.position = position
            else:
                new_unit = deepcopy(structural_unit)
                new_unit.position = position
                self.add_structural_unit(new_unit)

    def add_force_field(self, force_field, *interactions):

        """
        Adds a force field to *interactions.  If no interactions are
        passed, the force field is applied to all interactions in the universe.

        Arguments:
        force_field - the ForceField to be the interactions, or to all the
        interactions in the universe
        interactions - any objects with base class Interaction
        """

        self.force_fields = force_field

        if not interactions:
            self.force_fields.parameterize_interactions(self.interactions)
        else:
            self.force_fields.parameterize_interactions(*interactions)

    def add_bonded_interaction_pairs(self, *bonded_interaction_pairs):

        """
        Adds one or more interaction pairs to the universe

        Arguments:
        bonded_interaction_pairs - A list of (interaction, atoms) pairs, where
        atoms is a tuple of all atoms for that specific bonded interaction
        """

        self._bonded_interaction_pairs.update(bonded_interaction_pairs)

    def add_nonbonded_interaction(self, *nonbonded_interactions):

        """
        Adds one or more nonbonded interactions to the universe

        Arguments:
        nonbonded_interactions - a list of nonbonded interactions
        """

        self._nonbonded_interactions.update(nonbonded_interactions)


def _primitive_cubic(dimensions, number):

    """
    Generates a primitive cubic structure
    """

    raise NotImplementedError


def _liquid_structure():

    """
    Generates a random arrangement of structural units
    """

    raise NotImplementedError


class KSpaceSolver(object):

    """
    Class describing the k-space solver that is applied to electrostatic and/or
    dispersion interactions

    Attributes:
    solver - a string specifying the name of the solver. Solvers currently
    supported (although not by all MD engines):

    ewald
    PPPM
    """

    SOLVERS = ['ewald', 'pppm']

    def __init__(self, solver, **settings):

        """
        Different MD engines require different parameters to be specified for a
        k-space solver to be used. These parameters are specified in settings,
        which are grouped by engine.

        Arguments:
        solver - a string specifying the name of the solver

        Settings:
        LAMMPS:
        accuracy - a float specifying the relative RMS error in per-atom forces
        """

        self.solver = solver
        self.accuracy = settings.get('accuracy')

    @property
    def solver(self):

        return self._solver

    @solver.setter
    def solver(self, value):

        value = value.lower()
        if value not in self.__class__.SOLVERS:
            raise NotImplementedError('The solver type is not implemented for'
                                      ' any MD engine')
        self._solver = value


class ConstraintAlgorithm(object):

    """
    Class describing the algorithm and parameters which are applied to constrain
    bonded interactions

    Attributes:
    name - a string specifying the name of the constraint algorithm
    accuracy - a float specifying the accuracy (tolerance) of the applied
    constraints
    max_iterations - an integer specifying the maximum number of iterations that
    can be used when calculating the additional force that is required to
    constrain the atoms to satisfy the constraints on the bonded interactions
    """

    def __init__(self, accuracy, max_iterations):

        self.accuracy = accuracy
        self.max_iterations = max_iterations

    @property
    def name(self):

        return self.__class__.__name__

    @property
    def max_iterations(self):

        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, value):

        self._max_iterations = int(value)


class Shake(ConstraintAlgorithm):

    """
    Holds the parameters which are required for the SHAKE algorithm to be
    applied to the constrained interactions
    """

    def __init__(self, accuracy, max_iterations):

        super(Shake, self).__init__(accuracy, max_iterations)


class Rattle(ConstraintAlgorithm):

    """
    Holds the parameters which are required for the RATTLE algorithm to be
    applied to the constrained interactions
    """

    def __init__(self, accuracy, max_iterations):

        super(Rattle, self).__init__(accuracy, max_iterations)


class EnergyMinimizer(object):

    """
    Define the MD energy minimizer

    Attributes:
    n_steps - number of steps
    sz_steps - size of steps
    algorithm - minimization algorithm
    stop - condition for minimization to stop
    minimize() - uses MD engine API to minimize energy
    """

    def __init__(self):

        raise NotImplementedError

class Simulation(object):

    """
    Molecular dynamics engine for any ensemble

    Ensemble is defined by whether thermostats is included and a pressure is
    defined

    Attributes:
    universe - an MDMC universe with force field and atomic configuration
    engine - an instance of an external MD engine
    settings - a list of settings passed to MD engine, see __init__ for details
    trajectory - the trajectory generated by the MD simulation

    Methods:
    minimize - uses MD engine API to minimize the simulation energy
    run - uses MD engine API to run simulation
    """

    # TODO: Potentially separate out universe and simulation setup
    def __init__(self, universe, engine="mmtk", **settings):

        """
        Initializes universe, engine and settings

        Engine independent settings:
        temperature - float specifying simulation temperature in K
        time_step - float specifying simulation timestep size in fs
        integrator - string specifying MD time integrator

        Settings:
        lj_options - Options for Lennard-Jones interactions
        es_options - Options for electrostatic interactions
        thermostat - boolean defining if a thermostat is applied
        pressure - float specifying the pressure in units of Pa.  If this is
        defined then a barostat is applied.
        """

        self.universe = universe
        self.settings = settings
        self.engine = MDEngineFacadeFactory.create_facade(engine)
        self._setup()

    def _setup(self):

        """
        Creates a universe within the MD engine with the equivalent
        configuration and topology to self.universe and defines the simulation
        conditions
        """

        self.engine.setup_universe(self.universe, **self.settings)
        self.engine.setup_simulation(**self.settings)

    def minimize(self, n_steps):

        """
        Minimizes the MD simulation energy

        Arguments:
        n_steps - integer maximum number of steps to run the minimization
        """

        self.engine.minimize(n_steps)

    def run(self, n_steps, equilibration=False):

        """
        Runs the MD simulation

        Arguments:
        n_steps - an integer number of steps to run the simulation
        equilibration - boolean specifying if the run is for equilibration
        """

        self.engine.run(n_steps, equilibration)

    @property
    def trajectory(self):

        """
        Returns:
        MDMC trajectory calculated during the MD simulation run or None if no
        trajectory exists
        """

        try:
            return self.engine.convert_trajectory()
        except AttributeError:
            return None
