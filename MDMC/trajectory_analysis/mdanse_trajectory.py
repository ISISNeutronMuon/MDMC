#    This file is part of MDANSE.
#
#    MDANSE is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
from __future__ import annotations

from collections import ChainMap
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from MDANSE.Chemistry.ChemicalSystem import ChemicalSystem
from MDANSE.Framework.Units import measure
from MDANSE.MLogging import LOG
from MDANSE.MolecularDynamics.Configuration import (
    PeriodicRealConfiguration,
    RealConfiguration,
    _Configuration,
)
from MDANSE.MolecularDynamics.UnitCell import UnitCell
from MDANSE.Trajectory.FileTrajBase import TrajectoryFile

from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory

if TYPE_CHECKING:
    from collections.abc import Mapping

SLICE_ALL = np.s_[:]


def replace_isotope(atom_name: str):
    if ']' in atom_name:
        isotope_number, atom_symbol = atom_name.split(']')
        new_name = f"{atom_symbol}{isotope_number.strip('[]')}"
        return new_name
    return atom_name


class MDMCTrajectory(TrajectoryFile):
    """A wrapper around CompactTrajectory for MDANSE analysis."""

    UNIT_CONV = {
        "position": "nm",
        "box": "nm",
        "velocity": "nm/ps",
        "time": "ps",
    }

    UNIT_MDMC = {
        "position": "ang",
        "box": "ang",
        "velocity": "ang/fs",
        "time": "fs",
    }

    @property
    def chemical_system(self):
        if not self._chemical_system:
            from MDANSE.Chemistry.ChemicalSystem import ChemicalSystem
            cs_instance = ChemicalSystem("MDMC")
            new_element_list = [replace_isotope(atom) for atom in self.mdmc_trajectory.element_list]
            cs_instance.initialise_atoms(new_element_list)
            self._chemical_system = cs_instance
        return self._chemical_system

    def get_atom_property(self, atom_type: str, atom_property: str):
        self.chemical_system._database.get_atom_property(atom_type, atom_property)

    def __init__(self, mdmc_trajectory: CompactTrajectory):
        """Constructor.

        Parameters
        ----------
        h5_filename : Path or str
            The trajectory filename.
        """
        self.unit_cell_warning = ""

        self._h5_filename = "MDMC.traj"
        self._chemical_system = None
        self._h5_file = None

        self.mdmc_trajectory = mdmc_trajectory

        if self.chemical_system.rdkit_mol.GetNumBonds() > 0:
            configuration = self.configuration(0)
            contiguous_configuration = configuration.contiguous_configuration()
            coords = contiguous_configuration.coordinates
            self.chemical_system.set_bond_orders(coords)

    def _load_units(self) -> None:
        """Load units from h5 file."""
        self.unit_conv = {}
        self._units = {}

        for prop, target_unit in self.UNIT_CONV.items():
            self.unit_conv[prop] = measure(1.0, self.UNIT_MDMC[prop]).toval(target_unit)
            self._units[prop] = self.UNIT_MDMC[prop]

    def __len__(self) -> int:
        return len(self.mdmc_trajectory)

    @property
    def units(self) -> Mapping[str, str]:
        """Mapping of property labels to units."""
        return ChainMap(
            self._units,
            {"b_incoherent": "ang", "b_coherent": "ang"},
            self.chemical_system._database.units,
        )

    @classmethod
    def file_is_right(self, filename: Path | str) -> bool:
        """Check if the input file is likely to be an H5MD trajectory.

        Parameters
        ----------
        filename : Path | str
            File to check.

        Returns
        -------
        bool
            Whether file should be loaded as H5MD.

        """
        return True

    def close(self):
        """Close the trajectory."""

    def __getitem__(self, frame: int) -> dict[str, npt.NDArray[float]]:
        """Return the configuration at a given frame.

        Parameters
        ----------
        frame : int
            Frame to get.

        Returns
        -------
        dict[str, npt.NDArray[float]]
            Configuration at frame.
        """
        self._check_frame(frame)

        configuration = {
            "coordinates": self.mdmc_trajectory.positions[frame, :, :] * self.unit_conv["position"],
            "time": self.time()[frame],
        }

        if self.mdmc_trajectory.has_velocity:
            configuration["velocities"] = (
                self.mdmc_trajectory.velocities[frame, :, :] * self.unit_conv["velocity"]
            )

        configuration["unit_cell"] = self.unit_cell(frame)

        return configuration

    def charges(
        self,
        frame: int,
        indices: slice | int = np.s_[:],
    ) -> npt.NDArray[float]:
        """Return the electrical charge of atoms at a given frame.

        Parameters
        ----------
        frame : int
            Frame to load.

        Returns
        -------
        ndarray
            Charges at given time.

        """
        self._check_frame(frame)

        if isinstance(indices, int):
            indices = slice(indices, indices + 1)

        charge = self.mdmc_trajectory.atom_charges[frame, indices]

        return charge.astype(np.float64)

    def coordinates(
        self,
        frame: slice | int,
        atom_indices: slice | int = SLICE_ALL,
    ) -> npt.NDArray[float]:
        """Return the coordinates at a given frame.

        Parameters
        ----------
        frame : slice or int
            Frame(s) to load.
        atom_indices : slice or int
            Atoms to select.

        Returns
        -------
        ndarray
            The coordinates in given frame.

        """
        self._check_frame(frame)

        retval = self.mdmc_trajectory.positions[frame, atom_indices, :]

        return retval.astype(np.float64) * self.unit_conv["position"]

    def configuration(self, frame: int = 0) -> _Configuration:
        """Build and return a configuration at a given frame.

        Parameters
        ----------
        frame : int
            Frame to load.

        Returns
        -------
        _Configuration
            The configuration.

        """
        self._check_frame(frame)

        unit_cell = self.unit_cell(frame)

        variables = {}
        if self.mdmc_trajectory.has_velocity:
            self.variables["velocities"] = self.mdmc_trajectory.velocities

        coordinates = self.coordinates(frame)

        if unit_cell is None:
            conf = RealConfiguration(self._chemical_system, coordinates, **variables)
        else:
            conf = PeriodicRealConfiguration(
                self._chemical_system, coordinates, unit_cell, **variables,
            )

        return conf

    def time(self) -> npt.NDArray[float]:
        """Time timesteps from file."""

        return self.mdmc_trajectory.time * self.unit_conv["time"]

    def unit_cell(self, frame: int) -> UnitCell | None:
        """Return the unit cell at a given frame. If no unit cell is defined, returns None.

        Parameters
        ----------
        frame : int
            The frame number.

        Returns
        -------
        UnitCell or None
            The unit cell or None if no unit cells found.

        """
        return np.diag(self.mdmc_trajectory.dimensions) * self.unit_conv["box"]

    def masses(self) -> npt.NDArray[float]:
        """Get masses from databases.

        Parameters
        ----------
        atom_indices : Iterable[int] or slice or int
            Atoms to get masses for. (Default: all atoms)

        Returns
        -------
        npt.NDArray[float]
            Atomic masses.
        """
        return self.mdmc_trajectory.atom_masses.astype(np.float64)

    def has_variable(self, variable: str) -> bool:
        """Check if the trajectory has a specific variable e.g. velocities.

        Parameters
        ----------
        variable : str
            The variable to check the existence of.

        Returns
        -------
        bool
            True if variable exists.

        """
        return self.mdmc_trajectory.has_variable(variable)

    def atoms_in_database(self) -> list[str]:
        """Return the names of atoms defined in the atom property database.

        Here, it defaults to the central atom property database.

        Returns
        -------
        list[str]
            List of atom names that are present in the atom database.

        """
        return self.chemical_system._database.atoms

    @property
    def properties(self) -> list[str]:
        """Return the list of atom properties provided by the trajectory.

        Here, it defaults to the central atom property database.

        Returns
        -------
        list[str]
            List of atom property names that can be accessed.

        """
        return self.chemical_system._database.properties

    def variable(self, name: str) -> npt.NDArray[float]:
        """Return the dataset corresponding to a trajectory variable called 'name'."""
        return getattr(self.mdmc_trajectory, name, [])

    def variables(self) -> list[str]:
        """Return the names of available variables.

        Returns
        -------
        list[str]
            List of variables present in the file.

        """
        return ["velocities"] if self.mdmc_trajectory.has_velocity else []

    @property
    def has_velocity(self) -> bool:
        return self.mdmc_trajectory.has_velocity
