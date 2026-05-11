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
A module for writing and saving a H5MD file.
"""
import logging
from pathlib import Path

import h5py

from MDMC.common import units
from MDMC.trajectory_analysis.observables.obs import Observable

LOGGER = logging.getLogger(__name__)

def guess_unit(axis_label: str) -> str:
    """Return the physical unit of a dataset based on its text label."""
    match axis_label:
        case "Q":
            unit_string = units.SYSTEM["LENGTH"]**(-1)
        case "E":
            unit_string = units.SYSTEM["ENERGY_TRANSFER"]
        case "r":
            unit_string = units.SYSTEM["LENGTH"]
        case _:
            unit_string = "au"
            LOGGER.warning("MDA writer could not determine the unit of variable %s", axis_label)
    return unit_string.replace(" ","")

def write_metadata(target: h5py.File, observable: Observable):
    string_dt = h5py.special_dtype(vlen=str)
    meta_group = target.create_group("metadata")
    meta_group.create_dataset("task_name", (1,), data=observable.name, dtype=string_dt)
    meta_group.create_dataset(
        "MDMC_version",
        (1,),
        data="0.2.0",
        dtype=string_dt,
    )
    inputs_group = meta_group.create_group("inputs")
    inputs_group.create_dataset("output_files",
                                (1,),
                                data='["mdmc_output.mda", ["MDAFormat"], "no logs"]',
                                dtype=string_dt)

def write_MDA(observable: Observable,
              *,
              filename: str,
              file_loc: Path | str,
              timestamp: str,
              suffix: str = '.mda'):
    """Write the input observable to an MDANSE MDA file.

    Parameters
    ----------
    observable : Observable
        Object containing the calculated or experimental data.
    filename : str
        Base of the output file.
    file_loc : Path | str
        Path where the file will be written.
    timestamp : str
        Text to be used as the time stamp.
    suffix : str, optional
        File name extension, by default '.mda'.
    """
    obs_name = observable.name
    target_path = Path(file_loc, f"{filename}{timestamp}_{obs_name}").with_suffix(suffix)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(target_path, 'w') as target:
        result_group = target.create_group("mdmc_result")
        axes_group = result_group.create_group("axes")
        data_group = result_group.create_group(obs_name)
        for key, data in observable.independent_variables.items():
            temp_ds = axes_group.create_dataset(key, data=data)
            temp_ds.attrs["axis"] = "index"
            temp_ds.attrs["scaling_factor"] = 1.0
            temp_ds.attrs["units"] = guess_unit(key)
        for key, data in observable.dependent_variables.items():
            main_ds = data_group.create_dataset(key, data=data[0])
            main_ds.attrs["axis"] = "|".join(f"mdmc_result/axes/{x}"
                                               for x in observable.independent_variables)
            main_ds.attrs["scaling_factor"] = 1.0
            main_ds.attrs["units"] = "au"
            main_ds.attrs["tags"] = "main"
        write_metadata(target, observable)
