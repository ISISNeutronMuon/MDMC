# MDMC is a package for the optimisation of classical potentials with experimental data
# Copyright (C) 2026 MDMC Developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Module for memory-efficient MD trajectory handling.

Notes
-----
Seeing how the current (as of August 2022) trajectory implementation
consumes a lot of memory, an attempt is being made to build an object
that will contain the bare minimum of functionality, to show the
limits of performance that we can achieve within Python.
"""

# The main idea is that, ultimately, we need to store
# 3 * n_atoms * n_steps
# floating point numbers if we want to process a trajectory.
# Independent of the programming language we use, we can expect
# that each coordinate will be at least a 32-bit float, or
# a 64-bit float if we use the default value normally picked
# by numpy.
from typing import Any

import numpy as np

from MDMC.common import units
from MDMC.MD.structures import Atom
from MDMC.readers import H5MD_reader
from MDMC.trajectory_analysis.trajectory import TemporalConfiguration


class CompactTrajectory:
    """
    Store an MD trajectory in numpy arrays.

    The units are system units.

    The normal workflow for a CompactTrajectory is:

    1. Create and allocate memory.

       .. code-block:: python

          traj = CompactTrajectory(n_step, n_atoms, useVelocity)
          or
          traj = CompactTrajectory()
          traj.preAllocate(n_step, n_atoms, useVelocity)

    2. Write values into arrays.

       .. code-block:: python

          for n in frame_numbers:
              if not traj.validateTypes(atom_types):
                  traj.WriteOneStep(step_number = n, time = time_value, positions = pos_values)
                  traj.setDimensions(box_dimensions, step_number = n)

    3. Add chemical element information.

       .. code-block:: python

          traj.labelAtoms(atom_symbols, atom_masses)
          traj.setCharges(atom_charges)

    5. Check internal consistency, trim arrays.

       .. code-block:: python

          traj.postProcess()

    Parameters
    ----------
    n_steps : int, optional
        Number of simulation steps in the trajectory. Defaults to 0.

        ``n_steps = 0`` means that ``preAllocate`` will have to be run
        separately after the ``CompactTrajectory`` has been created.

        ``n_steps > 0`` means that ``preAllocate`` will be run immediately
        in the ``__init__`` function.
    n_atoms : int, optional
        Number of atoms in the system. Defaults to 1.
    useVelocity : bool, optional
        If the trajectory contains velocities, set to ``True``
        to allocate an additional array for the velocity values.
        Defaults to False.
    bits_per_number : int, optional
        Set number of bytes in floating point numbers.
    **settings : dict
        Extra options.
    """

    def __init__(
        self,
        n_steps: int = 0,
        n_atoms: int = 1,
        useVelocity: bool = False,
        bits_per_number: int = 8,
        **settings: Any,
    ):
        # The development plan is to use the units defined here to calculate conversion factors,
        # and use these factors when writing the numbers into the arrays.
        # For now, we define the units here:
        self.position_unit = units.SYSTEM["LENGTH"]
        self.time_unit = units.SYSTEM["TIME"]
        self.velocity_unit = units.SYSTEM["LENGTH"] / units.SYSTEM["TIME"]
        self.dtype = self._get_dtype(bits_per_number)  # this sets the data type to np.float64
        # The underlying assumption is that the number of atoms,
        # and the atom types, stay CONSTANT within the trajectory,
        # and so we define them as header data, and not separately for every frame:
        self.n_atoms = n_atoms
        self.n_steps = n_steps
        self.atom_types = []  # atom types defined following the MD engine naming
        self.atom_masses = []  # atom masses, floating point numbers, one for each atom
        self.atom_charges = []  # atom charges, floating point numbers, one for each atom
        # The initial value of self.dimensions is set to a number too low to be physical,
        # but different from 0. This way if an observable tried to calculate the Q vectors
        # or, more precisely, reciprocal space vectors
        # from an unpopulated CompactTrajectory, it would not divide by zero.
        self.dimensions = 0.1 * np.ones(3)  # avoids divide-by-zero errors, explained above.
        self.changing_dimensions = None
        self.element_list = []  # chemical element (str) defined for each atom
        self.element_set = set()  # set of chemical elements (str) present in the trajectory
        # key point: the data!
        # this is where we will keep the numpy arrays
        # vvvvvvvvvvvvvvvvvv
        self.position = None
        self.velocity = None
        self.time = None
        # now some state indicators:
        self.is_allocated = False  # preAllocate has been run, numpy arrays exist
        self.is_populated = False  # numpy arrays are not empty; some data have been written
        self.is_fixedbox = True  # the dimensions of the simulation box are fixed;
        self.has_velocity = useVelocity
        # ^^^^^^^^^^^^^^
        # the self.is_fixedbox could potentially be False for an NPT ensemble
        # in that case self.changing_dimensions will contain the box dimensions for each step.
        self.first_index = 1e5  # the first array index at which we have written data
        self.last_index = 0  # the last array index at which we have written data
        # ^^^^^^^^^^^^^
        # since we use np.empty, the elements of the array are _not_ initialised to any value,
        # so it is a precaution to slice the arrays at the end to cut off the empty parts.
        # -------------
        # now we allocate the arrays
        # vvvvvvvvvvvvv
        if n_steps > 0:
            self.preAllocate(n_steps=n_steps, n_atoms=n_atoms, useVelocity=useVelocity)
        try:
            self.universe = settings["universe"]
        except KeyError:
            self.universe = None

    @staticmethod
    def _get_dtype(bpn: int) -> np.dtype:
        """
        Get appropriate numpy type from bits per number.

        Parameters
        ----------
        bpn : int
            Bytes in floating point.

        Returns
        -------
        numpy.dtype
            Floating point datatype.
        """
        if bpn > 8:
            return np.float128
        if bpn > 4:
            return np.float64
        if bpn > 2:
            return np.float32
        return np.float16

    @staticmethod
    def create_from_h5md(file_name: str):
        """
        Create a CompactTrajectory from an H5MD file.

        Parameters
        ----------
        file_name : str
            Path to the H5MD file containing the desired trajectory.

        Returns
        -------
        CompactTrajectory
            The trajectory described by the file.
        """
        new_ct = CompactTrajectory()
        all_data = H5MD_reader.read_all_data(file_name)
        n_atoms = len(all_data["species"])
        n_steps = all_data["no_steps"]
        new_ct.preAllocate(n_atoms=n_atoms, n_steps=n_steps, useVelocity=True)
        new_ct.time = all_data["time"]
        new_ct.setCharge(all_data["charge"])
        new_ct.setDimensions(all_data["box_dimension"])
        new_ct.position = all_data["position"]
        new_ct.velocity = all_data["velocity"]
        new_ct.atom_masses = all_data["mass"]
        new_ct.atom_types = new_ct.validateTypes(all_data["atom_symbol"])
        new_ct.element_list = all_data["atom_symbol"]
        new_ct.element_set = list(set(all_data["atom_symbol"]))
        return new_ct

    def setBytesPerNumber(self, bytes_per_number: int = 8):
        """
        Change the number of bytes per number in the arrays.

        This is used for storing atom positions, velocities, the frame
        timestamps and simulation box dimensions.

        The best approach is to set the correct value before
        populating the arrays, but it is still possible to change the
        data type using this function when the ``CompactTrajectory``
        already contains some values.

        Parameters
        ----------
        bytes_per_number : int, optional
            If 8, the arrays will use :any:`np.float64` to store the
            positions, velocities and time at each step. Will always
            be rounded up to the nearest multiple of 2.

        See Also
        --------
        CompactTrajectory._get_dtype : Type getter.
        """
        self.dtype = self._get_dtype(bytes_per_number)
        if len(self) > 0:
            self.time = self.time.astype(self.dtype)
            self.position = self.position.astype(self.dtype)
            if self.velocity is not None:
                self.velocity = self.velocity.astype(self.dtype)
            self.changing_dimensions = self.changing_dimensions.astype(self.dtype)

    def __len__(self):
        if self.position is None:
            return 0
        # Since it is not guaranteed that the postProcess method has been run,
        # we explicitly limit the indices to those which have been written into.
        # The length corresponds to the number of simulation steps.
        return len(self.position[self.first_index : self.last_index + 1])

    def __getitem__(self, index: int | slice):
        # different behaviour:
        # a single index extracts a subtrajectory
        # with length 1,
        # while a slice with produce a subtrajectory,
        # which is another CompactTrajectory.
        try:
            start, stop, step = index.start, index.stop, index.step
        except AttributeError as exc:
            if index >= self.__len__():
                raise IndexError(
                    "Trying to access a nonexistent time frame in the CompactTrajectory.",
                ) from exc
            return self.subtrajectory(index, index + 1, 1)

        return self.subtrajectory(start, stop, step)

    def __eq__(self, other: "CompactTrajectory") -> bool:
        if not self.is_allocated == other.is_allocated:
            return False
        if not self.is_populated == other.is_populated:
            return False
        if not self.is_fixedbox == other.is_fixedbox:
            return False
        if not self.has_velocity == other.has_velocity:
            return False
        are_the_same = (
            other.position_unit == self.position_unit
            and other.time_unit == self.time_unit
            and other.velocity_unit == self.velocity_unit
            and other.dtype == self.dtype
        )
        if are_the_same:
            are_the_same = np.all(
                [
                    self.n_atoms == other.n_atoms,
                    self.first_index == other.first_index,
                    self.last_index == other.last_index,
                ],
            )
        else:
            print("Failed header comparison")
        if are_the_same:
            are_the_same = np.all(
                [
                    np.allclose(other.dimensions.shape, self.dimensions.shape),
                    np.allclose(other.position.shape, self.position.shape),
                    np.allclose(other.time.shape, self.time.shape),
                    np.allclose(other.changing_dimensions.shape, self.changing_dimensions.shape),
                ],
            )
        else:
            print("Failed index comparison")
        if are_the_same:
            are_the_same = self.universe == other.universe
        else:
            print("Failed shape comparison")
            print(f"Dimensions  other: {other.dimensions.shape}, self: {self.dimensions.shape}")
            print(f"Position    other: {other.position.shape}, self: {self.position.shape}")
            print(f"Time        other: {other.time.shape}, self: {self.time.shape}")
            print(
                f"CDim        other: {other.changing_dimensions.shape}"
                f"self: {self.changing_dimensions.shape}",
            )
        if are_the_same:
            if self.is_fixedbox:
                are_the_same = np.all(
                    [
                        np.allclose(other.dimensions, self.dimensions),
                        np.allclose(other.position, self.position),
                        np.allclose(other.time, self.time),
                    ],
                )
            else:
                are_the_same = np.all(
                    [
                        np.allclose(other.dimensions, self.dimensions),
                        np.allclose(other.position, self.position),
                        np.allclose(other.time, self.time),
                        np.allclose(other.changing_dimensions, self.changing_dimensions),
                    ],
                )
        else:
            print("Failed universe comparison")
        if are_the_same:
            if self.velocity is not None and other.velocity is not None:
                are_the_same = np.allclose(self.velocity, other.velocity)
        else:
            print("Failed array comparison")
        if are_the_same:
            are_the_same = np.all(
                [
                    np.all(other.atom_types == self.atom_types),
                    other.n_atoms == self.n_atoms,
                    np.allclose(other.atom_charges, self.atom_charges),
                    np.allclose(other.atom_masses, self.atom_masses),
                    other.element_list == self.element_list,
                    other.element_set == self.element_set,
                ],
            )
        else:
            print("Failed velocity comparison")
        if not are_the_same:
            print(f"atom_types: {np.all(other.atom_types == self.atom_types)}")
            print(f"n_atoms: {other.n_atoms == self.n_atoms}")
            print(f"atom_charges: {np.allclose(other.atom_charges, self.atom_charges)}")
            print(f"atom_masses: {np.allclose(other.atom_masses, self.atom_masses)}")
            print(f"element_list: {other.element_list == self.element_list}")
            print(f"element_set: {other.element_set == self.element_set}")
        return are_the_same

    @property
    def velocities(self) -> np.ndarray:
        """
        The velocity array.

        Added for those parts of code that use the plural form
        instead of singular.

        Returns
        -------
        ~numpy.ndarray
            Velocities array.
        """
        return self.velocity

    @property
    def times(self) -> np.ndarray:
        """
        The time array.

        Added for those parts of code that use the plural form
        instead of singular.

        Returns
        -------
        ~numpy.ndarray
            Time samples array.
        """
        return self.time

    @property
    def positions(self) -> np.ndarray:
        """
        The position array.

        Added for those parts of code that use the plural form
        instead of singular.

        Returns
        -------
        ~numpy.ndarray
            Time samples array.
        """
        return self.position

    @property
    def data(self) -> np.ndarray:
        """
        The step numbers and time steps in a single array.

        Returns
        -------
        ~numpy.ndarray
            Concatenation of separate data arrays.
        """
        return np.column_stack(
            [
                np.arange(self.n_steps),
                self.time,
                # there used to be a TemporalConfiguration here as well.
            ],
        )

    def preAllocate(self, n_steps: int = 1, n_atoms: int = 1, useVelocity: bool = False):
        """
        Allocate array space for data.

        Creates empty arrays for storing atom positions, velocities,
        time values of the simulation frames, and the simulation
        box dimensions.

        The numpy empty object do not initialise elements until they
        have been accessed by a read or write operation. This means
        that allocating too many steps is generally safe and harmless,
        as the unused steps will be removed when the postProcess
        method is called.

        Parameters
        ----------
        n_steps : int, optional
            Number of simulation steps in the
            trajectory. Defaults to 1.
        n_atoms : int, optional
            Number of atoms in the system.
            Defaults to 1.
        useVelocity : bool, optional
            If the trajectory contains
            velocities, set to `True` to allocate an additional array
            for the velocity values.
            Defaults to False.
        """
        if self.is_allocated:
            print("WARNING: preAllocate has already been run on this CompactTrajectory.")
        self.n_atoms = n_atoms  # from now on we decide that each step will have n_atoms
        self.n_steps = n_steps  # and there will be n_steps in this trajectory.
        # time steps are the first dimension of the arrays.
        shape = (n_steps, n_atoms, 3)
        self.time = np.empty(n_steps, dtype=self.dtype)
        self.position = np.empty(shape, dtype=self.dtype)
        if useVelocity:
            self.velocity = np.empty(shape, dtype=self.dtype)
        # First change of state: the arrays have been allocated.
        self.is_allocated = True
        self.changing_dimensions = np.empty((n_steps, 3), dtype=self.dtype)
        # ^^^^^^^^^^^^^^^^^^^^^^
        # We are fairly confident that the dimensions of the simulation box will NOT change
        # during the simulation, but allocating this array does not really cost us anything,
        # and, if it turns out that the dimensions do change, we will be prepared.

    def setCharge(self, charge_list: list[float] = None) -> bool:
        """
        Set the values of partial charge for each atom.

        It assumes that the charges are constant within
        the simulation.

        Parameters
        ----------
        charge_list : list[float]
            Fractional electrical charge of each atom,
            given in a list with one floating point number
            per atom.

        Returns
        -------
        bool
            Whether charges have been set.
        """
        # If need be, we can modify it to work the same
        # as setDimensions, so it populates a time-dependent
        # array if it turns out that the charge values
        # change between the time frames.
        if charge_list is None:
            return False
        if not len(charge_list) == self.n_atoms:
            raise ValueError("Wrong number of charges in setCharge.")
        self.atom_charges = np.array(charge_list)
        return True

    def setDimensions(self, frame_dimensions: np.ndarray = None, step_num: int = -1):
        """
        Write the simulation box dimensions into the object header.

        Additionally, if the simulation box dimensions change with time,
        it will keep track of the new dimensions at each step in the
        self.changing_dimensions object.

        Parameters
        ----------
        frame_dimensions : numpy.ndarray
            3 float numbers defining the size
            of the simulation box along x, y and z.
        step_num : int
            The number of the simulation frame at which
            the frame_dimensions array was read.
        """
        # The initial self.dimensions is set to 0.1 Angstrom.
        # This is too small to ever be used in a simulation, and at the same time
        # it is not 0.
        # Here we check if the initial values has already been overwritten with
        # a physical value of dimensions.
        if np.all(np.abs(self.dimensions - 0.1) < 1e-5):
            self.dimensions = frame_dimensions
            self.changing_dimensions[0] = frame_dimensions
        else:
            # we save the dimensions
            self.changing_dimensions[step_num] = frame_dimensions
            # per simulation step now.

    def writeOneStep(
        self,
        step_num: int = -1,
        time: float = -1.0,
        positions: np.ndarray = None,
        velocities: np.ndarray = None,
    ):
        """
        Write the atom properties of a single frame into the trajectory arrays.

        Parameters
        ----------
        step_num : int, optional
            The index at which the numbers will be written.
            Defaults to -1.
        time : float
            The time stamp of the simulation step, in the
            correct time units (femtoseconds). Defaults to -1.0.
        positions : np.ndarray
            The array of the atom positions, shaped
            (n_atoms, 3). Defaults to None.
        velocities : np.ndarray, optional
            The array of the atom velocities, shaped
            (n_atoms, 3). If we don't use velocities, it can be skipped.
            Defaults to None.
        """
        if not self.is_allocated:
            raise IndexError("Writing outside of the reserved array range.")
        self.time[step_num] = time
        self.position[step_num, :, :] = positions
        if velocities is not None:
            self.velocity[step_num, :, :] = velocities
        # some housekeeping:
        # we take note of the indices that have been written to.
        # In case we allocated more memory than needed,
        # we keep track of the indices where the data has been written,
        # so we can discard the unused part later,
        # as the uninitalised part of the array will contain random numbers.
        self.first_index = min(step_num, self.first_index)
        self.last_index = max(step_num, self.last_index)

    def writeEmptyStep(self, step_num: int = -1, time: float = -1.0):
        """
        Advance the iterators without writing any data.

        It is basically a reduced version of the writeOneStep method.
        This was added since the tests/trajectory_analysis/test_PDF.py
        use a trajectory made of 1000 _empty_ configurations.

        Parameters
        ----------
            step_num : int, optional
                The index at which the numbers will be written.
                Defaults to -1.
            time : float, optional
                The time stamp of the simulation step, in the
                correct time units (femtoseconds). Defaults to -1.0.
        """
        if not self.is_allocated:
            raise IndexError("Writing outside of the reserved array range.")
        self.time[step_num] = time
        self.first_index = min(step_num, self.first_index)
        self.last_index = max(step_num, self.last_index)

    def _create_types_from_elements(self, atom_types: list[str]) -> np.ndarray:
        """
        Translate IDs to elements.

        Parameters
        ----------
        atom_types : list[str]
            Mapping from from elements to integers.

        Returns
        -------
        numpy.ndarray
            List of new IDs.

        Notes
        -----
        Some engines (e.g. LAMMPS) like using numbers at atom types,
        while others use letters. CompactTrajectory was written for
        LAMMPS initially, and likes having numbers for types.
        This function will check if the types are numbers,
        and replace them with consitent positive numbers if it
        turns out that they were not.
        """
        if len(atom_types) == 0:
            return np.array([])
        try:
            int(atom_types[0])
        except ValueError:
            all_types = np.unique(atom_types)
            compare_types = np.array(atom_types, dtype=str)
            number_types = np.zeros(len(atom_types), dtype=np.int64)
            for number, at_type in enumerate(all_types, 1):
                number_types[np.where(compare_types == at_type)] = number
            return number_types
        return np.array(atom_types)

    def validateTypes(self, raw_atom_types: np.ndarray) -> bool:
        """
        Check and set the array of atom types.

        If previously run, verify the array of atom types from the new
        frame is the same as the original array of atom types.

        If the atom types have changed during the simulation,
        we cannot process the results using the ``CompactTrajectory``
        object, and the validation will return ``False``.
        It is up to the engine facade to decide what to do
        if ``validateTypes`` returns ``False``.

        Parameters
        ----------
        raw_atom_types : numpy.array
            An array of all the atom types, sorted by the atom ID.

            If these are numbers, like in a LAMMPS simulation,
            then they will be stored directly.

            If ``raw_atom_types`` are strings, an internal method
            called ``_create_types_from_elements`` will create
            numbers that correspond to the strings, so that
            every atom type has its own number.

        Returns
        -------
        bool
            Whether the atom types are unchanged.
        """
        # codes like LAMMPS make it possible to generate or destroy atoms/particles
        # in a simulation to simulate flow. It is unlikely to happen in an MDMC run.
        # Since we cannot handle such a case,
        # we will return False to indicate that this trajectory is not suitable for MDMC.
        atom_types = self._create_types_from_elements(raw_atom_types)
        # ^^^^^^^^
        # This ensures that the atom types are numers.
        if len(self.atom_types) == 0:  # case 1: atom_types have not been set
            if len(atom_types) == self.n_atoms:
                self.atom_types = atom_types  # and now they are set.
                return True
        elif np.all(self.atom_types == atom_types):  # case 2: atom_types have been set
            return True  # and have not changed
        return False  # case 3: atom_types have been set and have changed.

    def labelAtoms(
        self,
        atom_symbols: dict[int, str] = None,
        atom_masses: dict[int, float] = None,
    ) -> bool:
        """
        Create internal atom references.

        - A `list` of chemical elements of all the atoms in a trajectory frame,
        - a `set` of all the chemical elements present in a trajectory,
        - a `list` of atom masses of all the atoms in a trajectory frame.

        It is up to the engine facade to construct the input dictionaries containing
        the correct information about the chemical elements and masses.

        Parameters
        ----------
        atom_symbols : dict[int, str]
            Mapping from trajectory atom_ID to chemical symbol.
        atom_masses : dict[int, float]
            Mapping from trajectory atom_ID to chemical mass.

        Returns
        -------
        bool
            References created successfully.
        """
        if len(self.atom_types) == 0:
            return False  # tests/trajectory_analysis/test_PDF.py use empty Configurations
            # so we accept a case of no atoms in the CompactTrajectory.
            # We just return False in case we wanted to check in real code if
            # we are trying to set labels on a CompactTrajectory with no atoms.

        self.element_list = [str(atom_symbols[atom_id]) for atom_id in self.atom_types]
        self.element_set = set(self.element_list)
        self.atom_masses = np.array([atom_masses[atom_id] for atom_id in self.atom_types])
        return True

    def postProcess(self):
        """
        Finalise trajectory analysis.

        Notes
        -----
        This function should be called after the all the trajectory steps have
        been read. It will discard the unnecessary rows of the arrays,
        in case we had allocated too many due to some rounding error.
        """
        if not self.is_populated:
            self.position = self.position[self.first_index : self.last_index + 1]
            self.time = self.time[self.first_index : self.last_index + 1]
            if self.velocity is not None:
                self.velocity = self.velocity[self.first_index : self.last_index + 1]
            self.changing_dimensions = self.changing_dimensions[
                self.first_index : self.last_index + 1
            ]
            self.first_index = 0
            self.last_index = len(self.position) - 1
            dim_spread = np.std(self.changing_dimensions, axis=0)
            if np.allclose(dim_spread, 0.0):
                self.is_fixedbox = True
            else:
                self.is_fixedbox = False
                self.dimensions = self.changing_dimensions[
                    self.first_index : self.last_index + 1
                ].mean(
                    0,
                )
            self.is_populated = True

    def subtrajectory(
        self,
        start: int = 0,
        stop: int = -1,
        step: int = 1,
        atom_filter: list[int] = None,
    ):
        """
        Slice a ``CompactTrajectory``.

        Returns another ``CompactTrajectory`` instance, which contains
        the same header information, and a subset of the original
        trajectory steps.

        The arrays in the original trajectory will be sliced following
        the pattern: `new = old[start:stop:step]`.

        Optionally, ``atom_filter`` can be specified to choose only
        specific atoms from the trajectory.

        It is recommended not to use ``subtrajectory`` directly in
        this case, but to use the ``filter_by_element`` or
        ``filter_by_type`` methods, which will create the
        ``element_filter`` themselves.

        Parameters
        ----------
        start : int
            Number defining the beginning of the slicing range.
        stop : int
            Number defining the end of the slicing range.
        step : int
            Step size of the slicing operation.
        atom_filter : list[int]
            Atom indices to extract.

        Returns
        -------
        CompactTrajectory
            A trajectory containing the same header
            and the same or fewer steps than the original.

        See Also
        --------
        filter_by_element : Get components by species.
        filter_by_type : Get components by ID.
        """
        self.postProcess()
        temp = CompactTrajectory()
        # copy over all the transferable parts
        temp.position_unit = self.position_unit
        temp.time_unit = self.time_unit
        temp.velocity_unit = self.velocity_unit
        temp.dtype = self.dtype
        temp.dimensions = self.dimensions
        temp.universe = self.universe
        if atom_filter is None:
            temp.position = self.position[start:stop:step, :, :]
            temp.time = self.time[start:stop:step]
            temp.changing_dimensions = self.changing_dimensions[start:stop:step, :]
            if self.velocity is not None:
                temp.velocity = self.velocity[start:stop:step, :, :]
            temp.n_atoms = self.n_atoms
            temp.atom_types = self.atom_types.copy()
            temp.atom_masses = self.atom_masses.copy()
            temp.atom_charges = self.atom_charges.copy()
            temp.element_list = self.element_list
            temp.element_set = self.element_set
        else:
            temp.position = self.position[start:stop:step, atom_filter, :]
            temp.time = self.time[start:stop:step]
            temp.changing_dimensions = self.changing_dimensions[start:stop:step, :]
            if self.velocity is not None:
                temp.velocity = self.velocity[start:stop:step, atom_filter, :]
            temp.atom_types = [self.atom_types[x] for x in atom_filter]
            temp.n_atoms = len(temp.atom_types)
            temp.atom_charges = self.atom_charges[atom_filter]
            temp.atom_masses = self.atom_masses[atom_filter]
            temp.element_list = [self.element_list[x] for x in atom_filter]
            temp.element_set = set(temp.element_list)
        temp.is_allocated = True
        temp.is_populated = True
        temp.has_velocity = self.has_velocity
        temp.is_fixedbox = self.is_fixedbox
        temp.first_index = 0
        temp.last_index = len(temp.position) - 1
        temp.n_steps = len(temp.position)
        temp.postProcess()
        return temp

    def filter_by_time(self, start: float, end: float = None) -> "CompactTrajectory":
        """
        Filter to within time defined by ``start`` and ``end``.

        Create another CompactTrajectory, containing only the frames
        with the time values equal to or larger than the start parameter,
        and smaller than the end parameter.

        If only one time value is given, the function will create a
        CompactTrajectory with a single time step equal to start,
        or raise an error if start is not an element of the time array.

        Parameters
        ----------
        start : float
            The start time for filtering the ``Trajectory``.
        end : float, optional
            The end time for filtering the ``Trajectory``.  The default is
            `None`, which means the new returned ``Trajectory`` has a single
            time, defined by the ``start``.

        Returns
        -------
        CompactTrajectory
            A copy with ``times`` in half open interval defined by
            ``start`` and ``end``.

        Raises
        ------
        ValueError
            No valid MD frames found.
        """
        if end is None:
            index = np.where(self.time == start)[0].ravel()
            if not index.size:
                raise ValueError("The specified time range contains no MD frames")
            return self.subtrajectory(index[0], index[0] + 1)

        index = np.where((self.time >= start) & (self.time < end))[0].ravel()
        if not index.size:
            raise ValueError("The specified time range contains no MD frames")
        return self.subtrajectory(index[0], len(index))

    def filter_by_element(self, elements: list[str]) -> "CompactTrajectory":
        """
        Filter subtrajectory by chemical symbol.

        Parameters
        ----------
        elements : list[str]
            List of chemical elements given as string.

        Returns
        -------
        CompactTrajectory
            A ``CompactTrajectory`` containing only the atoms of
            the specified chemical elements.
        """
        indices = []

        for element in elements:
            if element in self.element_list:
                index = np.where(np.array(self.element_list) == element)[0].ravel()
                indices.append(index)
        index = np.concatenate(indices)
        index = np.sort(index)
        return self.subtrajectory(0, len(self), step=1, atom_filter=index)

    def filter_by_type(self, types: list[int | str]) -> "CompactTrajectory":
        """
        Filter subtrajectory by atom ID.

        Parameters
        ----------
        types : list[int|str]
            A list of atom IDs.

        Returns
        -------
        CompactTrajectory
            A ``CompactTrajectory`` containing only the atoms of
            the specified type.
        """
        type_set = {x for x in types}
        indices = [ind for ind, at_type in enumerate(self.atom_types) if at_type in type_set]
        return self.subtrajectory(0, len(self), step=1, atom_filter=indices)

    def exportAtom(self, step_number: int = 0, atom_number: int = 0):
        """
        Create an ``Atom`` object for a chosen time step of the simulation and atom number.

        Parameters
        ----------
        step_number : int
            Number of the simulation step at which the atom
            position should be read.
        atom_number : int
            Number of the atom in the trajectory that will
            be created as an ``Atom`` object.

        Returns
        -------
        Atom
            A single ``Atom`` object.

        Notes
        -----
        For compatibility with ``Trajectory``.
        """
        try:
            element = self.element_list[atom_number]
        except AttributeError:
            element = "?"

        try:
            velocity = self.velocity[step_number, atom_number, :]
        except AttributeError:
            velocity = (0.0, 0.0, 0.0)

        try:
            charge = self.atom_charges[atom_number]
        except IndexError:
            charge = 0.0

        return Atom(element, self.position[step_number, atom_number, :], velocity, charge=charge)


def configurations_as_compact_trajectory(
    *configs: list[TemporalConfiguration],
) -> CompactTrajectory:
    """
    Populate ``CompactTrajectory`` arrays from list of ``TemporalConfiguration``.

    Parameters
    ----------
    *configs : list[TemporalConfiguration]
        Most likely containing atoms.

    Returns
    -------
    CompactTrajectory
        With atom types and positions matching those in the input configurations.

    Raises
    ------
    TypeError
        No configurations provided.
    """
    if not configs:
        raise TypeError(
            "At least one Configuration is needed for the CompactTrajectory.fromConfigs()",
        )

    traj = CompactTrajectory(
        n_steps=len(configs),
        n_atoms=len(configs[0].atoms),
        useVelocity=len(configs[0].atom_velocities) > 0,
        universe=configs[0].universe,
    )

    for step_number, config in enumerate(configs):
        try:
            current_time = config.time
        except AttributeError:
            current_time = 0.0
        if len(config.data) > 0:
            atpos = np.vstack(config.atom_positions)
            atvel = np.vstack(config.atom_velocities)
            traj.writeOneStep(
                step_num=step_number,
                time=current_time,
                positions=atpos,
                velocities=atvel,
            )
        else:
            traj.writeEmptyStep(step_num=step_number, time=current_time)
        try:
            dim = config.universe.dimensions
        except AttributeError:
            continue
        else:
            traj.setDimensions(dim, step_num=step_number)
    # here we extract as much header information as possible
    # from the TemporalConfiguration
    elements = []  # list of 'chemical_element_symbol', 1 per atom
    # dictionary of 'chemical_element_symbol' : mass (float) in a.m.u.
    masses = {}
    # dictionary of 'atom_type' : mass (float) in a.m.u.
    mass_per_type = {}
    element_per_type = {}  # dictionary of 'atom_type' : 'chemical_element_symbol'
    types = []  # a list of 'atom_type', 1 entry for each atom
    id_values = []  # a list of 'atom_ID', 1 entry for each atom
    atom_charge = []
    # we assume -for now- that the Trajectory stores atom_type.
    has_types = True
    atom_counter = 0
    # we just iterate over Atom objects
    for nat, atom in enumerate(configs[0].atoms):
        element = atom.element.symbol
        mass = atom.mass
        charge = atom.charge
        elements.append(element)
        atom_charge.append(charge)
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
    traj.validateTypes(np.array(types)[np.argsort(id_values)])
    traj.labelAtoms(element_per_type, mass_per_type)
    traj.setCharge(atom_charge)
    traj.postProcess()
    return traj
