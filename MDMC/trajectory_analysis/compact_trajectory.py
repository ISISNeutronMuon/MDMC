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

from typing import Union
import numpy as np
from MDMC.common.units import Unit
from MDMC.MD.structures import Atom
from MDMC.trajectory_analysis.trajectory import TemporalConfiguration

class CompactTrajectory:
    """
    This is a _nearly_ drop-in replacement for the Trajectory object.
    The goal was to hold all the information we need using as little
    memory as possible. This way, even if we don't use this class
    directly in the code in the future, we can still use it as a
    benchmark reference.
    """
    def __init__(self, *configurations, **settings):
        """
        This is a bare constructor which initialises all the fields the basic trajectory
        will have.
        Args:
            configurations : TemporalConfiguration
            Any number of ``TemporalConfiguration`` objects can be passed to the constructor,
            to create the CompactTrajectory in the same manner as the old Trajectory.
            This is just an option added here for compatibility, and not to break the unit
            tests. Please refrain from using ``TemporalConfiguration`` objects, and
            just use the preAlloacte and writeOneStep methods."""
        # the idea is that all the numbers within the trajectory are given in the same units
        # and it will be our job to ensure that it is the case later in the code
        self.position_unit = Unit('Ang')
        self.time_unit = Unit('fs')
        self.velocity_unit = Unit('Ang')/Unit('fs')
        # some other information
        self.dtype = self._get_dtype(8) # this sets the data type to np.float64
        # the underlying assumption is that the number of atoms,
        # and the atom types, stay CONSTANT within the trajectory
        self.n_atoms = -1
        self.n_steps = 0
        self.atom_types = []
        self.atom_masses = []
        self.dimensions = 0.1*np.ones(3) # avoids divide-by-zero errors
        self.changing_dimensions = None
        self.element_list = []
        self.element_set = set()
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
        if len(configurations) > 0:
            self.fromConfigs(*configurations)
        try:
            self.universe = settings['universe']
        except KeyError:
            try:
                self.universe = configurations[0].universe
            except IndexError:
                self.universe = None

    @staticmethod
    def _get_dtype(bpn : int):
        if bpn > 8:
            return np.float128
        if bpn > 4:
            return np.float64
        if bpn > 2:
            return np.float32
        return np.float16

    def setBytesPerNumber(self, bytes_per_number: int = 8):
        """
        Changes the number of bytes per number in the arrays storing atom positions,
        velocities, the frame timestamps and simulation box dimensions.
        The best approach is to set the correct value before populating the arrays,
        but it is still possible to change the data type using this function
        when the CompactTrajectory already contains some numbers.
        Args:
            bytes_per_number (int, optional): If 8, the arrays will use np.float64 to
              store the positions, velocities and time at each step. Will always be rounded
              up to the nearest multiple of 2."""
        self.dtype = self._get_dtype(bytes_per_number)
        if len(self) > 0:
            self.times = self.times.astype(self.dtype)
            self.position = self.position.astype(self.dtype)
            if self.velocity is not None:
                self.velocity = self.velocity.astype(self.dtype)
            self.changing_dimensions = self.changing_dimensions.astype(self.dtype)

    def __len__(self):
        if self.position is None:
            return 0
        return len(self.position[self.first_index:self.last_index])

    def __getitem__(self, index: Union[int, slice]):
        try:
            start, stop, step = index.start, index.stop, index.step
        except AttributeError:
            return self.TemporalConfiguration(index)
        else:
            return self.subtrajectory(start, stop, step)

    @property
    def configurations(self):
        """
        This is a bit of a hack, really. The code frequently uses
        trajectory.configurations[0].universe, and such, but here
        we store the header in the trajectory itself.
        With this property, trajectory.configurations[0] evaluates
        to trajectory.
        """
        return [self]

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

    def fromConfigs(self, *configs): # Pylint didn't like the type tip here.
        """
        Populate the arrays of the CompactTrajectory using the input list
        of ``TemporalConfiguration`` objects.
        This method has been added to increase the compatibility between
        the ``Trajectory`` and ``CompactTrajectory``.
        """
        # print(f"In fromConfigs: received {len(configs)} configurations.")
        # print(configs[0])
        if len(configs) < 1:
            raise TypeError("At least one Configuration is needed"
                                " for the CompactTrajectory.fromConfigs()")
        self.preAllocate(n_steps = len(configs),
                         n_atoms = len(configs[0].atoms),
                         useVelocity = len(configs[0].atom_velocities) > 0)
        for n, c in enumerate(configs):
            if len(c.data) > 0:
                atpos = np.row_stack(c.atom_positions)
                atvel = np.row_stack(c.atom_velocities)
                self.writeOneStep(step_num = n,
                                time = c.time,
                                positions = atpos,
                                velocities = atvel)
            else:
                self.writeEmptyStep(step_num = n,
                                time = c.time)
            try:
                dim = c.universe.dimensions
            except AttributeError:
                continue
            else:
                self.setDimensions(dim, step_num = n)
        # here a lengthy section of the code that is supposed to
        # fill in the missing information
        # if we are loading a pickled Trajectory
        # instead of reading a proper trajectory file.
        elements = [] # list of 'chemical_element_symbol', 1 per atom
        masses = {} # dictionary of 'chemical_element_symbol' : mass (float) in a.m.u.
        mass_per_type = {} # dictionary of 'atom_type' : mass (float) in a.m.u.
        element_per_type = {} # dictionary of 'atom_type' : 'chemical_element_symbol'
        types = [] # a list of 'atom_type', 1 entry for each atom
        id_values = [] # a list of 'atom_ID', 1 entry for each atom
        has_types = True # we assume -for now- that the Trajectory stores atom_type.
        atom_counter = 0
        for nat, atom in enumerate(configs[0].atoms): # we just iterate over Atom objects
            element = atom.element
            mass = atom.mass
            elements.append(element)
            masses[element] = mass
            if has_types:
                try:
                    types.append(atom.atom_type)
                except AttributeError:
                    has_types = False
            id_values.append(atom.ID)
            atom_counter = nat + 1
        if not has_types:
            all_elements = sorted(list(np.unique(elements)))
            types = [all_elements.index(x) for x in elements]
        for x in range(atom_counter):
            mass_per_type[types[x]] = masses[elements[x]]
            element_per_type[types[x]] = elements[x]
        self.validateTypes(np.array(types)[np.argsort(id_values)])
        self.labelAtoms(element_per_type, mass_per_type)
        self.postProcess()

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
        if np.all(np.abs(self.dimensions - 0.1) < 1e-5):
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

    def writeEmptyStep(self, step_num: int = -1, time: float = -1.0):
        """
        This function advances the iterators without writing any data.
        Again, it was added to make the unit tests work.
        Args:
            step_num (int, optional): the index at which the numbers will be written.
               Defaults to -1.
            time (float, optional): the time stamp of the simulation step, in the
               correct time units (femtoseconds). Defaults to -1.0.
        """
        if not self.is_allocated:
            pass # here I should use a fallback function later
        else:
            self.times[step_num] = time
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
            types, sorted by the atom ID. These are supposed
            to be numbers, like in a LAMMPS simulation.

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
        if len(self.atom_types) == 0:
            # raise AttributeError("labelAtoms: atom_types have not been set.")
            return False # tests use empty trajectories - can't raise an exception
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
        temp.n_steps = len(temp.position)
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

        index = np.where(self.times==start)[0].ravel()
        if end is None:
            if len(index) < 1:
                raise ValueError("Start is not in self.times")
            return self.subtrajectory(index[0],index[0]+1)
        total = np.where(np.logical_and(self.times >= start, self.times < end))[0].ravel()
        if len(total) < 1:
            raise ValueError("The specified time range contains no MD frames")
        return self.subtrajectory(index[0], len(total))

    def Atom(self, step_number: int = 0, atom_number: int = 0):
        """
        For compatibility with ``Trajectory``, creates an ``Atom`` object
        for a chosen time step of the simulation and atom number.

        Parameters
        ----------
        step_number : int
            number of the simulation step at which the atom
            position should be read
        atom_number : int
            number of the atom in the trajectory that will
            be created as an ``Atom`` object

        Returns
        -------
        Atom
            a single ``Atom`` object, for whatever purpose the user
            may need it.
        """
        try:
            element = self.element_list[atom_number]
        except AttributeError:
            element = '?'
        try:
            velocity = self.velocity[step_number, atom_number, :]
        except AttributeError:
            velocity = (0.0, 0.0, 0.0)
        # :TODO: some information about charge is needed
        return Atom(element,
            self.position[step_number, atom_number, :],
            velocity)

    def TemporalConfiguration(self, step_number: int = 0) -> TemporalConfiguration:
        """
        For compatibility, creates a TemporalConfiguration object
        out of the numpy arrays of atom positions.

        Parameters
        ----------
        step_number : int
            The ``TemporalConfiguration`` will be created using the
            positions and time from this step of the ``CompactTrajectory``

        Returns
        -------
        TemporalConfiguration
            A ``TemporalConfiguration`` object containing all the atoms at the
            requested time step
        """
        return TemporalConfiguration(self.times[step_number],
            *[self.Atom(step_number, at_num) for at_num in range(self.n_atoms)],
            universe = self.universe)
