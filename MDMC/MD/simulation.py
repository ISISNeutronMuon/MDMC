"""Module for setting up and running the simulation

 Classes for the simulation box, minimizer and integrator.

 AUTHOR :    Thomas Farmer        START DATE :    2018-4-30 13:01:04"""

from copy import deepcopy
import itertools

from enum import Enum
import numpy as np

from MDMC.MD.engine_facades.facade_factory import MDEngineFacadeFactory
from MDMC.trajectory_analysis.trajectory import Configuration


Shape = Enum('Shape', ['cubic', 'orthorhombic', 'infinite',
                       'rhombic_dodecahedron', 'truncated_octahedron'])


class Universe(object):

    """
    Class where configuration and topology are defined

    Attributes:
    shape - member of the Shape enum
    dims - array of dimensions
    """

    def __init__(self, dimensions, shape=Shape.cubic, force_field=None,
                 structures=None):

        """
        Arguments:
        dimensions - single float for cubic universe or 3 element list of floats
        shape - member of shape enum
        """

        self.shape = shape
        self.dims = dimensions
        if structures:
            self.configuration = Configuration(structures)
        else:
            self.configuration = Configuration(universe=self)
        self._interactions = set()
        self.force_fields = force_field

    @property
    def dims(self):

        return self._dims

    @dims.setter
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

        return self._interactions

    @property
    def parameters(self):

        return set([param for interaction in self.interactions
                    for param in interaction.params])

    @property
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

    def add_structural_unit(self, structural_unit, force_field=None):

        """
        Adds a single structural unit to the universe, with optional force field
        applying only to that structural unit
        """

        structural_unit.universe = self
        self.configuration.add_structural_unit(structural_unit)
        for atom in structural_unit.atom_list:
            self._interactions.update(atom.interactions)
        if force_field:
            self.add_force_field(force_field, structural_unit.interactions)

    # TODO: Add in option to tessellate a configuration to fill universe (a la GROMACS)
    def fill(self, structural_unit, force_field=None,
             structural_motif=_liquid_structure(), **kwargs):

        """
        A liquid-like filling of the universe independent of existing atoms

        Adds copies of structural_unit to existing configuration until universe
        is full.  As exclusion region is defined by the size of a bounding
        sphere, this method is most suitable for atoms or molecules with
        approximately equal dimensions.

        CURRENT APPROACH RESULTS IN NUMBER DENSITY DIFFERENT TO WHAT IS
        SPECIFIED DEPENDING ON HOW CLOSE CUBE ROOT OF N_MOLECULES IS TO AN INT.

        Args:
        structural_unit
        number_density - in AA^-3
        forcefield - Simultaneously applies a forcefield
        structural_motif - The arrangement of structural units
        """

        # TODO: implement method for specifying number of molecules and number density, rather than box size
        n_units_xyz = self.dims / (1. / kwargs.get('num_density')) ** (1. / 3.)
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

        positions = sorted(list(itertools.product(*positions)))

        # TODO: See if changing from appending to extending improves performance
        for position in positions:
            if position is positions[0]:
                self.add_structural_unit(structural_unit)
                structural_unit.position = position
                if force_field:
                    self.add_force_field(force_field,
                                         structural_unit.interactions)
            else:
                new_unit = deepcopy(structural_unit)
                new_unit.position = position
                self.add_structural_unit(new_unit)

    def add_force_field(self, force_field, *interactions):

        """
        Adds a force field to the passed interactions.  If no interactions are
        passed, the force field is applied to all interactions in the universe.
        """

        if not interactions:
            self.force_fields = force_field(self.interactions)
        else:
            self.force_fields = force_field(*interactions)


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

class NVESimulation(object):

    """
    Molecular dynamics engine for NVE ensemble

    Attributes:
    universe - an MDMC universe with force field and atomic configuration
    engine - an instance of an external MD engine
    settings - a list of settings passed to MD engine, see __init__ for details

    Methods:
    run() - uses MD engine API to run simulation
    """

    # TODO: Potentially separate out universe and simulation setup
    def __init__(self, universe, engine="mmtk", **settings):

        """
        Initializes universe, engine and settings

        Engine independent settings:
        temperature - simulation temperature in K
        time_step - simulation timestep size in fs
        integrator - MD time integrator

        MMTK specific settings:
        lj_options - Options for Lennard-Jones interactions
        es_options - Options for electrostatic interactions
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
        """

        self.engine.minimize(n_steps)

    def run(self, n_steps, equilibration=False):

        """
        Runs the MD simulation
        """

        self.engine.run(n_steps, equilibration)

    @property
    def trajectory(self):

        """
        Returns:
        MDMC trajectory calculated during the MD simulation run
        """

        return self.engine.convert_trajectory()
