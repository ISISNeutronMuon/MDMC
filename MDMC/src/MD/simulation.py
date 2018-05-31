"""Module for setting up and running the simulation

 Classes for the simulation box, minimizer and integrator.

 AUTHOR :    Thomas Farmer        START DATE :    2018-4-30 13:01:04"""

from enum import Enum
import itertools
from copy import deepcopy
import numpy as np
import weakref

from MDMC.src.MD.engine_facades.facade_factory import MDEngineFacadeFactory
from MDMC.src.MD.structural_units import Molecule
from MDMC.src.trajectory_analysis.trajectory import Configuration

Shape = Enum('Shape',['cubic','orthorhombic','infinite','rhombic_dodecahedron',
                'truncated_octahedron'])

# TODO: Extract out atomic structure generation from Universe class, so that more structures can be easily added
# TODO: Atomic Structures should be factory

def _primitive_cubic(dimensions,number):
    pass

def _liquid_structure():
    pass

class Universe(object):

    """
    Class where configuration and topology are defined

    DESCRIPTION

    Attributes:
    shape
    dims - dimensions
    pbc - Periodic Boundary Conditions
    """

    def __init__(self,dimensions,shape=Shape.cubic):
        # TODO: Change interactions so that it maintains an ordered set, so that searching is optimized
        self.dims = np.array(dimensions)
        self.shape = shape
        self.configuration = Configuration()
        self._interactions = set()
        self.force_field = None

    @property
    def interactions(self):
        return self._interactions

    @property
    def volume(self):
        return np.prod(self.dims)

    # TODO: Potentially remove duplicate option and just rely on fill method
    def add_structural_unit(self,structural_unit,duplicates=0,
                            structure='liquid',force_field=None):
        self._add_single_structural_unit(structural_unit)
        if force_field:
            self.add_force_field(force_field,structural_unit.interaction_set())
        # TODO: Extract below into fill method
        # Calculate what bounding radius factor needs to be so structural units
        # are approximately equidistant and pass this to fill.
        for _ in range(duplicates):
            self.configuration.add_structural_units(deepcopy(structural_unit))

    def _add_single_structural_unit(self,structural_unit):
        self.configuration.add_structural_units(structural_unit)
        structural_unit.universe = weakref.ref(self)
        for atom in structural_unit.atom_list:
            self._interactions.update(atom.interaction_set())

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

        Raises:
        EXCEPTIONS
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
                self._add_single_structural_unit(structural_unit)
                offset = (structural_unit.position - structural_unit.bounding_box.min)
                structural_unit.position = position + offset
                if force_field:
                    self.add_force_field(force_field,structural_unit.interaction_set())
            else:
                new_unit = deepcopy(structural_unit)
                new_unit.position = position + offset
                self.configuration.add_structural_units(new_unit)

    # TODO: Add duplicate structural unit method

    # TODO: Extract this code into another method so that atoms is not regenerated each time
    def atom_list(self):
        return self.configuration.atom_list

    def add_force_field(self, force_field, *interactions):
        if not interactions:
            interactions = (self.interactions,)
        self.force_field = force_field(*interactions)

    @property
    def element_list(self):
        return [atom.element for atom in self.atom_list()]

    @property
    def element_dict(self):

        """
        Returns a dictionary of all elements and a single atom of that
        element type. This is required for MD engines which assign the same
        potential parameters for all identical element types.
        """

        return {atom.element:atom for atom in self.atom_list()}

    # TODO: Determine method of getting molecule list with looser coupling
    @property
    def molecule_list(self):
        return self.configuration.molecule_list


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
