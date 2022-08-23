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
    """
    This is a _nearly_ drop-in replacement for the Trajectory object.
    The goal was to hold all the information we need using as little
    memory as possible. This way, even if we don't use this class
    directly in the code in the future, we can still use it as a
    benchmark reference.
    """
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
        self.changing_dimensions = None
        self.element_list = []
        self.element_set = {}
        # key point: the data!
        # this is where we will keep the numpy arrays
        self.position = None
        self.velocity = None
        self.times = None
        # now some state indicators
        self.is_allocated = False
        self.is_populated = False
        self.is_fixedbox = True
        self.first_index = 0
        self.last_index = -1

    @staticmethod
    def _get_dtype(bpn : int):
        if bpn > 8:
            return np.float128
        if bpn > 4:
            return np.float64
        if bpn > 2:
            return np.float32
        return np.float16

    def __len__(self):
        if self.position is None:
            return 0
        return len(self.position[self.first_index:self.last_index])

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
        self.is_allocated = True
        self.changing_dimensions = np.empty((n_steps,3), dtype = self.dtype)

    def setDimensions(self, frame_dimensions: np.array = None,
                      step_num: int = -1):
        """
        Writes the simulation box dimensions into the object header.
        Additionally, if the simulation box dimensions change with time,
        it will keep track of the new dimensions at each step in the
        self.changing_dimensions object.

        Args:
            frame_dimensions (np.array): 3 float numbers defining the size
            of the simulation box along x, y and z.

            step_num (int): The number of the simulation frame at which
            the frame_dimensions array was read.
        """
        if np.all(np.abs(self.dimensions) < 1e-5):
            self.dimensions = frame_dimensions
        elif np.allclose(frame_dimensions, self.dimensions, rtol = 1e-6, atol = 1e-4):
            pass
        else:
            self.is_fixedbox = False
            self.changing_dimensions[step_num] = frame_dimensions

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
            self.last_index = max(step_num + 1, self.last_index)

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
        elif np.all(self.atom_types == atom_types):
            return True
        return False

    def labelAtoms(self, atom_symbols: dict = None, atom_masses: dict = None):
        """
        Populates the self.element_list with the correct atom symbols
        based on the list of atom_types from the trajectory file.
        Also, populates the self.element_set, which will then contain
        all the chemical elements present in the trajectory.

        Args:
            atom_symbols (dict): a dictionary which returns a 'str'
            chemical element symbol for every 'int' atom_ID from
            the LAMMPS trajectory.

            atom_masses (dict): a dictionary of 'float' atom masses,
            using as keys the 'int' atom_ID values from LAMMPS.

        """
        self.element_list = [atom_symbols[xx] for xx in self.atom_types]
        self.element_set = set(self.element_list)
        self.atom_masses = [atom_masses[xx] for xx in self.atom_types]

    def postProcess(self):
        """
        This function can be called after the all the trajectory steps have
        been read. It will discard the unnecessary rows of the arrays,
        in case we had allocated too many due to some rounding error.
        """
        if not self.is_populated:
            self.position = self.position[self.first_index:self.last_index]
            self.times = self.times[self.first_index:self.last_index]
            if self.velocity is not None:
                self.velocity = self.velocity[self.first_index:self.last_index]
            self.first_index = 0
            self.last_index = len(self.position)
            self.is_populated = True

    def subtrajectory(self, start: int = 0, stop: int = -1, step: int = 1):
        """
        Returns another CompactTrajectory instance, which contains the
        same header information, and a subset of the original trajectory
        steps. The arrays in the original trajectory will be sliced
        following the pattern: new = old[start:stop:step]

        Args:
            start (int): number defining the beginning of the slicing range.

            stop (int): number defining the end of the slicing range.

            step (int): step size of the slicing operation

        Returns:
            CompactTrajectory: a trajectory containing the same header
            and the same or less steps than the original.
        """
        self.postProcess()
        temp = CompactTrajectory()
        # copy over all the transferable parts
        temp.position_unit = self.position_unit
        temp.time_unit = self.time_unit
        temp.velocity_unit = self.velocity_unit
        temp.dtype = self.dtype
        temp.n_atoms = self.n_atoms
        temp.atom_types = self.atom_types
        temp.atom_masses = self.atom_masses
        temp.dimensions = self.dimensions
        temp.element_list = self.element_list
        temp.element_set = self.element_set
        # key point: the data!
        # this is where we will keep the numpy arrays
        temp.position = self.position[start:stop:step, : , :]
        temp.times = self.times[start:stop:step]
        if self.velocity is not None:
            temp.velocity = self.velocity[start:stop:step, : , :]
        # now some state indicators
        temp.is_allocated = True
        temp.is_populated = True
        temp.first_index = 0
        temp.last_index = len(temp.position)
        return temp
    def filter_by_time(self, start, end=None):
        """
        Filter the ``CompactTrajectory`` by time.
        Added only for compatibility with the original ``Trajectory``.

        Parameters
        ----------
        start : float
            The start time for filtering the ``Trajectory``
        end : , optional
            The end time for filtering the ``Trajectory``.  The default is
            `None`, which means the new returned ``Trajectory`` has a single
            time, defined by the ``start``

        Returns
        -------
        CompactTrajectory
            A ``CompactTrajectory`` with ``times`` in half open interval defined by
            ``start`` and ``end``
        """

        index = np.where(self.times==start).ravel()
        if end is None:
            if len(index) < 1:
                raise ValueError("Start is not in self.times")
            return self.subtrajectory(index[0],index[0]+1)
        total = np.where(np.logical_and(self.times >= start, self.times < end)).ravel()
        if len(total) < 1:
            raise ValueError("The specified time range contains no MD frames")
        return self.subtrajectory(index[0], len(total))

        