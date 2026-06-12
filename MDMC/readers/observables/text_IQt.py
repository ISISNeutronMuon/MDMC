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

"""Reader for multiple text files."""

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from MDMC.common import units

logger = logging.getLogger(__name__)

axes_target_units = {
    "r": units.SYSTEM["LENGTH"],
    "omega": units.SYSTEM["ENERGY_TRANSFER"],
    "romega": units.SYSTEM["ENERGY_TRANSFER"],
    "Q": units.SYSTEM["LENGTH"] ** -1,
    "time": units.SYSTEM["TIME"],
}


def read_xye(
    filename: str,
    x_index: int = 0,
    y_index: int = 1,
    e_index: int = 2,
    separator: str = " ",
    comment: str = "#",
) -> dict[str, np.typing.NDArray[np.floating]]:
    total_array = []
    needed_len = max(x_index, y_index, e_index) + 1
    with open(filename, "r") as source:
        for line in source:
            toks = line.split(comment)[0].split(separator)
            if len(toks) < needed_len:
                continue
            total_array.append([float(x) for x in toks])
    total_array = np.array(total_array)
    return {
        "x": total_array[:, x_index],
        "y": total_array[:, y_index],
        "e": total_array[:, e_index],
    }


class text_IQt:
    """
    Reads data from CSV files.
    """

    def __init__(
        self,
        filename_list: Sequence[Sequence[str, float]],
        xye_indices: tuple[int, int, int] = (0, 1, 2),
        in_file_axis: tuple[str, str] = ("time", "10 ^ -9 s"),
        other_axis: tuple[str, str] = ("Q", "Ang ^ -1"),
        separator: str = " ",
        comment: str = "#",
        variable_name: str = "IQt",
    ):
        self._independent_variables = {}
        self._dependent_variables = {}
        self._errors = {}
        self.x_index, self.y_index, self.e_index = xye_indices
        self.axis1, self.axis1_unit = in_file_axis[0], units.Unit(in_file_axis[1])
        self.axis2, self.axis2_unit = other_axis[0], units.Unit(other_axis[1])
        self._filename_list = filename_list
        self._sep_char = separator
        self._comm_char = comment
        self.variable_name = variable_name

    def parse(
        self,
        axis1_limits: tuple[float, float] | None = None,
        axis2_limits: tuple[float, float] | None = None,
        **settings: Any,
    ) -> None:
        """Read data from the input file."""
        data_array, error_array, axis2_array = [], [], []
        axis1_array = None
        for filename, ax2val in self._filename_list:
            if axis2_limits is not None and (
                ax2val < min(*axis2_limits) or ax2val > max(*axis2_limits)
            ):
                continue
            axis2_array.append(ax2val)
            file_contents = read_xye(
                filename,
                x_index=self.x_index,
                y_index=self.y_index,
                e_index=self.e_index,
                comment=self._comm_char,
                separator=self._sep_char,
            )
            if axis1_limits is not None:
                mask = np.logical_and(
                    file_contents["x"] >= min(*axis1_limits),
                    file_contents["x"] <= max(*axis1_limits),
                )
                ax1 = file_contents["x"][mask]
                ys1 = file_contents["y"][mask]
                es1 = file_contents["e"][mask]
            else:
                ax1 = file_contents["x"]
                ys1 = file_contents["y"]
                es1 = file_contents["e"]
            if axis1_array is None:
                axis1_array = ax1
            elif not np.allclose(axis1_array, ax1):
                raise ValueError(f"Axis1 in file {filename} does not match the previous files")
            data_array.append(ys1)
            error_array.append(es1)
        data_array = np.hstack([arr[:, None] for arr in data_array])
        error_array = np.hstack([arr[:, None] for arr in error_array])
        axis2_array = np.array(axis2_array)
        self._independent_variables = {
            self.axis1: axis1_array * self.axis1_unit.conversion_factor,
            self.axis2: axis2_array * self.axis2_unit.conversion_factor,
        }
        self._dependent_variables[self.variable_name] = data_array
        if self.e_index is not None:
            self._errors[self.variable_name] = error_array
        else:
            self._errors[self.variable_name] = np.sqrt(data_array)

    @property
    def independent_variables(self) -> dict:
        """
        Get the independent variables, Q (in ``Ang^-1``) and E (``meV``)

        Returns
        -------
        dict
            The independent variables Q and E
        """

        return self._independent_variables

    @property
    def dependent_variables(self) -> dict:
        """
        Get the dependent variables, SQw (in ``arb``)

        Returns
        -------
        dict
            The dependent variables, SQw (in ``arb``)
        """

        return self._dependent_variables

    @property
    def errors(self) -> dict:
        """
        Get the errors on the dependent variables

        Returns
        -------
        dict
            The error on SQw (in ``arb``)
        """

        return self._errors
