"""Module for setting up and running the simulation

 Classes for the simulation box, minimizer and integrator.

 AUTHOR :    Thomas Farmer        START DATE :    2018-4-30 13:01:04"""

from enum import Enum
import itertools
from copy import deepcopy
import numpy as np

from MDMC.src.MD.engine_facades.facade_factory import MDEngineFacadeFactory
from MDMC.src.trajectory_analysis.trajectory import Configuration

Shape = Enum('Shape', ['cubic', 'orthorhombic', 'infinite',
                        'rhombic_dodecahedron', 'truncated_octahedron'])

# TODO: Extract out atomic structure generation from Universe class, so that more structures can be easily added

def _primitive_cubic(dimensions, number):

    pass

def _liquid_structure():

    pass

class Universe(object):

    """
    Class where configuration and topology are defined

    Attributes:
    shape
    dims - dimensions
    pbc - Periodic Boundary Conditions
    """

    def __init__(self, dimensions, shape=Shape.cubic, force_field=None,
        structures=None):

        # TODO: Change interactions to maintain an ordered set, so that searching is optimized
        self.dims = np.array(dimensions)
        self.shape = shape
        if structures:
            self.configuration = Configuration(structures)
        else:
            self.configuration = Configuration()
        self._interactions = set()
        self.force_fields = force_field

    @property
    def interactions(self):

        return self._interactions

    @property
    def volume(self):

        return np.prod(self.dims)

    def add_structural_unit(self,structural_unit, force_field=None):

        """
        Adds a single structural unit to the universe, with optional force field
        applying only to that structural unit
        """

        self.configuration.add_structural_units(structural_unit)
        structural_unit.universe = self
        for atom in structural_unit.atom_list:
            self._interactions.update(atom.interactions)
        if force_field:
            self.add_force_field(force_field, structural_unit.interactions)

    # TODO: Add in option to tessellate a configuration to fill universe (a la GROMACS)
    def fill(self,structural_unit,force_field=None,
                structural_motif=_liquid_structure(),**kwargs):

        """
        A fluid-like filling of the universe independent of existing atoms

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
        n_units_xyz = self.dims/(1./kwargs.get('num_density'))**(1./3.)
        n_units_xyz = n_units_xyz.astype(int)

        positions = []
        for i in range(len(self.dims)):
            positions.append(np.linspace(0,self.dims[i],n_units_xyz[i],
                endpoint = False))

        positions = sorted(list(itertools.product(*positions)))

        # TODO: See if changing from appending to extending improves performance
        for position in positions:
            if position is positions[0]:
                self.add_structural_unit(structural_unit)
                offset = (structural_unit.position - structural_unit.bounding_box.min)
                structural_unit.position = position + offset
                if force_field:
                    self.add_force_field(force_field,structural_unit.interactions)
            else:
                new_unit = deepcopy(structural_unit)
                new_unit.position = position + offset
                self.configuration.add_structural_units(new_unit)

    @property
    def atom_list(self):
        return self.configuration.atom_list

    # TODO: Change this so that multiple forcefields can be stored - this will necessitate a change in how force fields are passed to MMTK
    def add_force_field(self, force_field, *interactions):

        """
        Adds a force field to the passed interactions.  If no interactions are
        passed, the force field is applied to all interactions in the universe.
        """

        if not interactions:
            self.force_fields = force_field(self.interactions)
        else:
            self.force_fields = force_field(*interactions)

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
    def molecule_list(self):

        return self.configuration.molecule_list

    # TODO: Implement
    def interaction_filter(self, condition):

        """
        Generic method for filtering interactions
        """

        raise NotImplementedError

    # TODO: Add helper methods for common filtering operations

# TODO: Implement
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

# TODO: Create other ensembles by decorating NVESimulation with thermo/barostats
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

        self.engine.setup_universe(self.universe,**self.settings)
        self.engine.setup_simulation(self.universe,**self.settings)

    def run(self,n_steps):
        self.engine.run(n_steps)
