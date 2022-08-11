"""Module for minimalistic trajectory handling.
Seeing how the current (as of August 2022) trajectory implementation
consumes a lot of memory, an attempt is being made to build an object
that will contain the bare minimum of functionality, to show the
limits of performance that we can achieve within Python.

The main idea is that, ultimately, we need to store
3 * n_atoms * n_steps
floating point numbers if we want to process a trajectory.
Independent of the programming language we use, we can expect
that each coordinate will be at least a 32-bit float, or
a 64-bit float if we use the default value normally picked
by numpy.
"""

import numpy as np
from MDMC.common.units import Unit

class CompactTrajectory:
    def __init__(self):
        """
        This is a bare constructor which initialises all the fields the basic trajectory
        will have."""
        # the idea is that all the numbers within the trajectory are given in the same units
        # and it will be our job to ensure that it is the case later in the code
        self.position_unit = Unit('Ang')
        self.time_unit = Unit('fs')
        self.velocity_unit = Unit('Ang')/Unit('fs')
        # some other information
        # the underlying assumption is that the number of atoms,
        # and the atom types, stay CONSTANT within the trajectory
        self.n_atoms = -1
        self.n_steps = 0
        self.atom_types = []
        # key point: the data!
        # this is where we will keep the numpy arrays
        self.position = None
        self.velocity = None
        self.time = None
        # now some state indicators
        self.is_allocated = False
        self.is_populated = False
        self.first_index = 0
        self.last_index = -1
    def preAllocate(self, n_steps=1,n_atoms=1, useVelocity = False):
        """
        In the case of 
        Args:
            n_steps (int, optional): _description_. Defaults to 1.
            n_atoms (int, optional): _description_. Defaults to 1.
            useVelocity (bool, optional): _description_. Defaults to False.
        """
    def takeOneStep(self, atom_ids: np.array, atom_positions: np.array):
        """
        The idea is to take one step of the simulation, and sort the atom
        positions according to atom numbers
        Args:
            atom_ids (np.array): _description_
            atom_positions (np.array): _description_
        """
        