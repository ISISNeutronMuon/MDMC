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
    def __init__(self, bytes_per_number: int = 8):
        """
        This is a bare constructor which initialises all the fields the basic trajectory
        will have.
        Args:
            bytes_per_number (int, optional): If 8, the arrays will use np.float64 to
              store the positions, velocities and time at each step. Will always be rounded
              up to the nearest multiple of 2."""
        # the idea is that all the numbers within the trajectory are given in the same units
        # and it will be our job to ensure that it is the case later in the code
        self.position_unit = Unit('Ang')
        self.time_unit = Unit('fs')
        self.velocity_unit = Unit('Ang')/Unit('fs')
        # some other information
        self.dtype = self._get_dtype(bytes_per_number)
        # the underlying assumption is that the number of atoms,
        # and the atom types, stay CONSTANT within the trajectory
        self.n_atoms = -1
        self.n_steps = 0
        self.atom_types = []
        self.atom_masses = []
        self.dimensions = np.zeros(3)
        # key point: the data!
        # this is where we will keep the numpy arrays
        self.position = None
        self.velocity = None
        self.times = None
        # now some state indicators
        self.is_allocated = False
        self.is_populated = False
        self.first_index = 0
        self.last_index = -1
    @staticmethod
    def _get_dtype(bpn : int):
        if bpn > 8:
            return np.float128
        elif bpn > 4:
            return np.float64
        elif bpn > 2:
            return np.float32
        else:
            return np.float16
    def preAllocate(self, n_steps: int = 1, n_atoms: int = 1,
                    useVelocity: bool = False):
        """
        For the best performance, we should already know how many
        steps the trajectory has, and how many atoms are in it.
        Then we can allocate the arrays immediately, and save ourselves
        the overhead of increasing the size of the data step by step.
        Args:
            n_steps (int, optional): Number of simulation steps in the
              trajectory. Defaults to 1.
            n_atoms (int, optional): Number of atoms in the system.
              Defaults to 1.
            useVelocity (bool, optional): If the trajectory contains
              velocities, set to True to allocate an additional array
              for the velocity values. Defaults to False.
        """
        self.n_atoms = n_atoms
        self.n_steps = n_steps
        shape = (n_steps, n_atoms, 3)
        self.times = np.empty(n_steps, dtype = self.dtype)
        self.position = np.empty(shape, dtype = self.dtype)
        if useVelocity:
            self.velocity = np.empty(shape, dtype = self.dtype)
    def writeOneStep(self, step_num: int = -1, time: float = -1.0,
                     positions: np.array = None,
                     velocities: np.array = None):
        """
        This function assumes that we have allocated the memory already,
        and that we have sorted the atoms according to their IDs.
        It will then put the numbers into the arrays at the correct index
        Args:
            step_num (int, optional): the index at which the numbers will be written.
               Defaults to -1.
            time (float, optional): the time stamp of the simulation step, in the
               correct time units (femtoseconds). Defaults to -1.0.
            positions (np.array, optional): the array of the atom positions, shaped
               (n_atoms, 3). Defaults to None.
            velocities (np.array, optional): the array of the atom velocities, shaped
               (n_atoms, 3). If we don't use velocities, it can be skipped.
               Defaults to None.
        """
        if not self.is_allocated:
            pass # here I should use a fallback function later
        else:
            self.times[step_num] = time
            self.position[step_num, : , :] = positions
            if velocities is not None:
                self.velocity[step_num, :, :] = velocities
            # some housekeeping:
            # we take note of the indices that have been written to.
            # Just in case the simulation was cut short, we will know
            # how many elements we can still use
            self.first_index = min(step_num, self.first_index)
            self.last_index = max(step_num, self.last_index)
    def validateTypes(self, atom_types: np.array):
        """This function checks if the sorted array of atom types
        from the new frame is the same as the original array
        of atom types.
        If the atom types have changed during the simulation,
        we cannot process the results using the CompactTrajectory
        object, and the validation will return False

        Args:
            atom_types (np.array): an array of all the atom
            types, sorted by the atom ID.

        Returns:
            bool: True if the atom types are the same as in
            the beginning, False otherwise.
        """
        if len(self.atom_types) == 0:
            if len(atom_types) == self.n_atoms:
                self.atom_types = atom_types
                return True
            else:
                return False
        else:
            if np.all(self.atom_types == atom_types):
                return True
            else:
                return False

        