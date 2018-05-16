"""Module for setting up and running the simulation

 Classes for the simulation box, minimizer and integrator.

 AUTHOR :    Thomas Farmer        START DATE :    2018-4-30 13:01:04"""

from MDMC.src.MD.MMTK.MMTK_wrapper import MMTK

from enum import Enum
import itertools
from copy import deepcopy
import numpy as np
import weakref

Shape = Enum('Shape','orthorhombic')
Boundary = Enum('Boundary', 'infinite cubic orthorhombic')

# TODO: Extract out atomic structure generation from Universe class, so that more structures can be easily added
# TODO: Atomic Structures should be factory

def _primitive_cubic(dimensions,number):
    pass

def _liquid_structure():
    pass

class Universe(object):
    """Class where configuration and topology are defined

    DESCRIPTION

    Attributes:
    shape
    dims - dimensions
    pbc - Periodic Boundary Conditions"""

    def __init__(self,dimensions,shape=Shape.orthorhombic,pbc=Boundary.cubic):
        # TODO: Change interactions so that it maintains an ordered set, so that searching is optimized
        self.dims = np.array(dimensions)
        self.shape = shape
        self.pbc = pbc
        self.configuration = []
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
            add_force_field(force_field,structural_unit.interaction_set())
        # TODO: Extract below into fill method
        # Calculate what bounding radius factor needs to be so structural units
        # are approximately equidistant and pass this to fill.
        for _ in range(duplicates):
            self.configuration.append(deepcopy(structural_unit))

    def _add_single_structural_unit(self,structural_unit):
        self.configuration.append(structural_unit)
        structural_unit.universe = weakref.ref(self)
        for atom in structural_unit.atom_list:
            self._interactions.update(atom.interaction_set())

    # TODO: Add in option to tessellate a configuration to fill universe (a la GROMACS)
    def fill(self,structural_unit,force_field=None,
                structural_motif=_liquid_structure(),**kwargs):
        """A fluid-like filling of the universe independent of existing atoms

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
        EXCEPTIONS"""

        self._add_single_structural_unit(structural_unit)
        structural_unit.position = [0,0,0]
        if force_field:
            self.add_force_field(force_field,structural_unit.interaction_set())

        # Calculate number of structural units required to achieve number_density
        # Create a generator to determine the position of each structural_unit in the given structural_motif
        # Create each structural_unit with a deepcopy and then set position

        # TODO: implement method for specifying number of molecules and number density, rather than box size
        num_units = int(self.volume / kwargs['num_density']) - 1
        unit_separation = (self.dims / int((num_units**(1./3.))))
        '''positions is sliced so that [0,0,0] is excluded'''
        # TODO: Replace arange with linspace as arange is unreliable with non-integer steps
        positions = sorted(list(itertools.product(
                            np.arange(0,self.dims[0],unit_separation[0]),
                            np.arange(0,self.dims[1],unit_separation[1]),
                            np.arange(0,self.dims[2],unit_separation[2]))))[1:]
        # TODO: See if changing from appending to extending improves performance
        for position in positions:
            new_unit = deepcopy(structural_unit)
            new_unit.position = position
            self.configuration.append(new_unit)

    # TODO: Add duplicate structural unit method

    # TODO: Extract this code into another method so that atoms is not regenerated each time
    def atom_list(self):
        atoms = []
        for structure in self.configuration:
            atoms.extend(structure.atom_list)
        return atoms

    def add_force_field(self,force_field,*interactions):
        if not interactions:
            interactions = (self.interactions,)
        self.force_field = force_field(*interactions)


class EnergyMinimizer(object):
    """Define the MD energy minimizer

     Attributes:
     n_steps - number of steps
     sz_steps - size of steps
     algorithm - minimization algorithm
     stop - condition for minimization to stop
     minimize() - uses MD engine API to minimize energy
     """

# TODO: Create other ensembles by decorating NVESimulation with thermo/barostats
class NVESimulation(object):
    """Molecular dynamics engine for NVE ensemble

    Attributes:
    time_step - time per step
    integrator - time integrators
    temperature - simulation temperature in K
    run() - uses MD engine API to run simulation
    """

    def __init__(self,time_step,temperature,
                    integrator="velocity_verlet",**settings):
        self.time_step = time_step
        self.temperature = temperature
        self.integrator = integrator
        self.settings = settings

    def run(self,engine=MMTK,n_steps):
        engine(self.time_step,self.temperature,self.integrator,
                self.settings)
