"""
A module for writing and saving a H5MD file.
"""
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from MDMC.common import units
from MDMC.trajectory_analysis.observables.obs import Observable


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
            raise ValueError(f"Could not guess the unit of data axis {axis_label}")
    return unit_string.replace(" ","")

def write_metadata(target: h5py.File, observable: Observable):
    string_dt = h5py.special_dtype(vlen=str)
    meta_group = target.create_group("metadata")
    meta_group.create_dataset("task_name", (1,), data=observable.name, dtype=string_dt)
    meta_group.create_dataset(
        "MDMC_version",
        (1,),
        data=str("0.2.0"),
        dtype=string_dt,
    )
    inputs_group = meta_group.create_group("inputs")
    inputs_group.create_dataset("output_files",
                                (1,),
                                data='["mdmc_output.mda", ["MDAFormat"], "no logs"]',
                                dtype=string_dt)

def write_MDA(observable: Observable,
              filename: str,
              file_loc: Path | str,
              timestamp: bool = True,
              suffix: str = '.mda'):
    target_path = (Path(file_loc) / (str(filename) + "_result")).with_suffix(suffix)
    with h5py.File(target_path, 'w') as target:
        obs_name = observable.name
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
            main_ds.attrs["axis"] = "|".join(["mdmc_result/axes/"+str(x) for x in observable.independent_variables])
            main_ds.attrs["scaling_factor"] = 1.0
            main_ds.attrs["units"] = "au"
        write_metadata(target, observable)
