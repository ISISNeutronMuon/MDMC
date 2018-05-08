"""Module for setting up and running the simulation

 Classes for the simulation box, minimizer and integrator.

 AUTHOR :    Thomas Farmer        START DATE :    2018-4-30 13:01:04"""

from enum import Enum

Shape = Enum('Shape','orthorhombic')
Boundary = Enum('Boundary', 'infinite cubic orthorhombic')

class Universe(object):
    """Class where configuration and topology are defined

    DESCRIPTION

    Attributes:
    shape
    dims - dimensions
    pbc - Periodic Boundary Conditions"""

    def __init__(self,dimensions,shape=Shape.orthorhombic,pbc=Boundary.cubic):
        self.dims = dimensions
        self.shape = shape
        self.pbc = pbc
        self.configuration = []
        self._interactions = set()

    def add_structure(self,structural_unit):
        self.configuration.append(structural_unit)
        structural_unit.universe = self
        for atom in structural_unit.atom_list():
            self._interactions.update(atom.interaction_set())

    # TODO: Extract this code into another method so that atoms is not regenerated each time
    def atom_list(self):
        atoms = []
        for structure in self.configuration:
            atoms.extend(structure.atom_list())
        return atoms

    def interaction_set(self):
        return self._interactions

    def add_force_field(self,force_field):
        

    # TODO: Implement ForceField on universe
    """When a ForceField object is created and applied to the universe, the
    universe assigns interaction types and strengths to all interactions."""


class EnergyMinimizer(object):
    """Define the MD energy minimizer

     Attributes:
     n_steps - number of steps
     sz_steps - size of steps
     algorithm - minimization algorithm
     stop - condition for minimization to stop
     minimize() - uses MD engine API to minimize energy
     """


class NVESimulation(object):
    """Molecular dynamics engine for NVE ensemble

     Decorated with other ensembles which require thermostats/barostats.

     Attributes:
     n_steps - number of steps
     time_step - time per step
     integrator - time integrators
     temperature - simulation temperature in K
     run() - uses MD engine API to run simulation
     """
