"""An adapter which will allow a CompactTrajectory instance
existing in the memory to be used by MDANSE.
MDANSE normally uses an HDF5 file for trajectory storage,
since this allows only the relevant frames to be loaded
into the memory, as needed.
However, in the case of MDMC we expect that the normal
operation will rely on shorter trajectories, and the
additional step of writing the trajectory again as
and HDF5 file will cause an unnecessary performance loss.
"""

# The main idea is that, ultimately, we need to store
# 3 * n_atoms * n_steps
# floating point numbers if we want to process a trajectory.
# Independent of the programming language we use, we can expect
# that each coordinate will be at least a 32-bit float, or
# a 64-bit float if we use the default value normally picked
# by numpy.

import numpy as np
from MDANSE.Chemistry.ChemicalEntity import ChemicalSystem, Atom
from MDANSE.MolecularDynamics.Configuration import PeriodicRealConfiguration
from MDANSE.MolecularDynamics.UnitCell import UnitCell
from MDANSE.Extensions import atomic_trajectory

from MDMC.common import units
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory


class MdanseTrajectory:
    """Wrapper class around the MDMC CompactTrajectory to make it work
    like an MDANSE Trajectory instance in MDANSE.

    It converts the units of position and time from MDMC units to MDANSE
    units, and builds the ChemicalSystem out of atom information.
    """

    def __init__(self, mdmc_trajectory: CompactTrajectory):
        self.mdanse_time_unit = units.Unit("ps")
        self.mdanse_length_unit = units.Unit("nm")

        self.mdmc_trajectory = mdmc_trajectory
        self.chemical_system = ChemicalSystem("MDMC trajectory")
        self._unit_cells = [None] * len(mdmc_trajectory)

        self.populate_chemical_system()

    def populate_chemical_system(self):
        """Transfers the necessary information from MDMC CompactTrajectory
        to the MDANSE ChemicalSystem."""

        for atom_number in np.arange(self.mdmc_trajectory.n_atoms):
            chemical_element = self.mdmc_trajectory.element_list[atom_number]
            atom_name = "_".join([chemical_element, str(atom_number + 1)])
            mdanse_atom = Atom(chemical_element, name=atom_name)
            self.chemical_system.add_chemical_entity(mdanse_atom)

        # MDANSE Trajectory in Trajectory.py sets the configuration in the ChemicalSystem
        # to contain the unit cell and atom positions from the first simulation frame.
        # This may be important, so we do the same thing here.
        cell = self.mdmc_trajectory.dimensions
        cell_3x3 = np.zeros((3, 3))
        cell_3x3[0, 0] = cell[0]
        cell_3x3[1, 1] = cell[1]
        cell_3x3[2, 2] = cell[2]
        unit_cell = UnitCell(cell_3x3 / self.mdanse_length_unit.conversion_factor)
        coords = self.mdmc_trajectory.position[0]
        conf = PeriodicRealConfiguration(self.chemical_system, coords, unit_cell)
        self.chemical_system.configuration = conf

    @property
    def timestep(self):
        """Calculates the time step of the MDMC trajectory, and converts it to
        the MDANSE time units.

        Returns:
            float: time step value
        """
        if len(self.mdmc_trajectory) > 1:
            timeaxis = self.mdmc_trajectory.time
            tstep = (timeaxis[1:] - timeaxis[:-1]).mean()
        else:
            tstep = 1.0
        return tstep / self.mdanse_time_unit.conversion_factor

    @property
    def has_velocity(self) -> bool:
        """True if velocities are stored in the trajectory, False otherwise.

        Returns:
            bool - True or False value if the velocities are saved in the trajectory.
        """
        return self.mdmc_trajectory.has_velocity

    def read_atomic_trajectory(
        self,
        index: int,
        first: int = 0,
        last: int = None,
        step: int = 1,
        box_coordinates: bool = False,
    ) -> np.ndarray:
        """Extracts the trajectory of a single atom from the trajectory object.

        Arguments:
            index -- index of the atom for which the trajectory will be produced

        Keyword Arguments:
            first -- index of the first trajectory frame (default: {0})
            last -- index of the last trajectory frame (default: {None})
            step -- step size in trajectory frames (default: {1})
            box_coordinates -- flag: use fractional coordinates? (default: {False})

        Returns:
            np.ndarray containing the trajectory of a single atom
        """

        if last is None:
            last = len(self)

        subtrajectory = self.mdmc_trajectory.subtrajectory(first, last, step, [index])
        coords = np.squeeze(subtrajectory.position.astype(np.float64))
        unit_cells = [self.unit_cell(x) for x in range(first, last, step)]

        direct_cells = np.array([uc.transposed_direct for uc in unit_cells])
        inverse_cells = np.array([uc.transposed_inverse for uc in unit_cells])
        atomic_traj = atomic_trajectory.atomic_trajectory(
            coords, direct_cells, inverse_cells, box_coordinates
        )
        return atomic_traj

    def read_configuration_trajectory(
        self,
        index: int,
        first: int = 0,
        last: int = None,
        step: int = 1,
        variable: str = "velocities",
    ) -> np.ndarray:
        """Extracts the information (i.e. velocity) for a single atom out of the trajectory.

        Arguments:
            index -- index of the atom for which the velocities will be produced

        Keyword Arguments:
            first -- index of the first trajectory frame (default: {0})
            last -- index of the last trajectory frame (default: {None})
            step -- step size in trajectory frames (default: {1})
            box_coordinates -- flag: use fractional coordinates? (default: {False})

        Raises:
            AttributeError: If velocities are not present in the trajectory
            NotImplementedError: if something else than velocities is needed

        Returns:
            np.ndarray of velocities of a single atom
        """
        if variable == "velocities":
            if not self.has_velocity:
                raise AttributeError("Trajectory does not contain velocities")
        else:
            raise NotImplementedError("Only velocities are supported")

        temp_trajectory = self.mdmc_trajectory.subtrajectory(
            first, last, step, atom_filter=[index]
        )

        result = temp_trajectory.velocity / (
            self.mdanse_length_unit.conversion_factor
            / self.mdanse_time_unit.conversion_factor
        )

        return result

    # Here we add the methods that exist only for compatibility with
    # MDANSE MolecularDynamics.Trajectory.Trajectory class

    def coordinates(self, frame: int) -> np.ndarray:
        """Returns atom coordinates at a specified simulation frame,
        in MDANSE units (nm).

        Arguments:
            frame (int) -- number of simulation step

        Raises:
            IndexError: if requested index is out of range

        Returns:
            np.ndarray -- an (N, 3) array of atom positions (in nm)
        """

        if frame < 0 or frame >= len(self):
            raise IndexError("Invalid frame number")
        coords = self.mdmc_trajectory.position[frame]
        return coords / self.mdanse_length_unit.conversion_factor

    def unit_cell(self, frame: int) -> "UnitCell":
        """Returns an MDANSE UnitCell for a specific frame number.

        Arguments:
            frame (int) -- number of the simulation step

        Raises:
            IndexError: if the frame is out of range

        Returns:
            UnitCell -- the unit cell definition for frame
        """
        if self._unit_cells[frame] is not None:
            return self._unit_cells[frame]
        if frame < 0 or frame >= len(self):
            raise IndexError("Invalid frame number")

        if self.mdmc_trajectory.is_fixedbox:
            cell = self.mdmc_trajectory.dimensions
        else:
            cell = self.mdmc_trajectory.changing_dimensions[frame]
        cell_3x3 = np.zeros((3, 3))
        cell_3x3[0, 0] = cell[0]
        cell_3x3[1, 1] = cell[1]
        cell_3x3[2, 2] = cell[2]
        result = UnitCell(cell_3x3 / self.mdanse_length_unit.conversion_factor)
        self._unit_cells[frame] = result
        return result

    def configuration(self, frame: int) -> PeriodicRealConfiguration:
        """Builds and returns an MDANSE PeriodicRealConfiguration
        for the specified simulation step.

        Arguments:
            frame (int) -- number of the simulation step

        Returns:
            PeriodicRealConfiguration - a snapshot of the system,
                containing chemical description, coordinates and unit cell definition
        """
        coordinates = self.coordinates(frame)
        unit_cell = self.unit_cell(frame)

        conf = PeriodicRealConfiguration(self.chemical_system, coordinates, unit_cell)

        return conf

    def close(self):
        """We don't need to close any files"""

    def __len__(self):
        return len(self.mdmc_trajectory)

    def __getitem__(self, frame_number: int) -> np.ndarray:
        return self.coordinates(frame_number)
