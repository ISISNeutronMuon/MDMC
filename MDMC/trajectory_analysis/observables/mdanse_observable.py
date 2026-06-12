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

"""Wrapper for analysis run using MDANSE."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import numpy.typing as npt
from MDANSE.Framework.Configurators.IConfigurator import IConfigurator
from MDANSE.Framework.InstrumentResolutions.IInstrumentResolution import IInstrumentResolution
from MDANSE.Framework.Jobs.IJob import IJob
from MDANSE.Framework.Units import measure
from MDANSE.IO.IOUtils import summarise_array
from more_itertools import first

from MDMC.trajectory_analysis.observables.obs import Observable
from MDMC.trajectory_analysis.observables.obs_factory import ObservableFactory

if TYPE_CHECKING:
    from pathlib import Path

    from MDMC.MD.simulation import Universe
    from MDMC.readers.observables.csv_reader import csv_reader


job_aliases = {
    "SQw": "NeutronDynamicTotalStructureFactor",
    "PDF": "PairDistributionFunction",
}

ENERGY_MDANSE_TO_MDMC = measure(1.0, iunit="rad/ps", equivalent=True).toval("meV")
Q_MDANSE_TO_MDMC = measure(1.0, iunit="1/nm").toval("1/ang")
R_MDANSE_TO_MDMC = measure(1.0, iunit="nm").toval("ang")
TIME_MDANSE_TO_MDMC = measure(1.0, iunit="ps").toval("fs")

if hasattr(IInstrumentResolution, "indirect_subclasses"):
    MDANSE_RESOLUTION_FUNCTIONS = IInstrumentResolution.indirect_subclasses()
else:
    MDANSE_RESOLUTION_FUNCTIONS = IInstrumentResolution.available_classes()


def run_ndtsf_special_case(MD_input, file_path: Path | None = None, verbose=0, **parameters):
    """Evaluate the function using the current parameter values.

    Gets the current values of parameters from trajectory attributes.

    Parameters
    ----------
    MD_input : CompactTrajectory
        An empty trajectory with a parameters attribute.
    verbose : int, optional
        Ignored, by default 0.
    """
    output_temp_files = {
        job_type: NamedTemporaryFile(delete=False)  # noqa: SIM115
        for job_type in [
            "DynamicCoherentStructureFactor",
            "DynamicIncoherentStructureFactor",
        ]
    }
    for job_type, file_object in output_temp_files.items():
        output_filename = file_object.name
        file_object.close()
        job_instance = IJob.create(job_type)
        settings = get_default_mdanse_settings(job_type)
        settings["instrument_resolution"] = ("ideal", {})
        settings["frames"] = [
            0,
            len(MD_input),
            1,
            parameters.get("correlation_frames", len(MD_input) // 2),
        ]
        for key, value in parameters.items():
            settings[key] = value
        if q_shells := parameters.get("q_shells"):
            settings["q_vectors"][1]["shells"] = q_shells
            settings["q_vectors"][1]["width"] = q_shells[-1]
        settings["trajectory"] = file_path
        settings["output_files"] = [output_filename, ["MDAFormat"], "no logs"]
        job_instance.setup(settings)
        job_instance.run(settings, status=True)
    job_instance = IJob.create("NeutronDynamicTotalStructureFactor")
    settings = get_default_mdanse_settings(job_type)
    settings["trajectory"] = file_path
    settings["dcsf_input_file"] = output_temp_files["DynamicCoherentStructureFactor"].name + ".mda"
    settings["disf_input_file"] = (
        output_temp_files["DynamicIncoherentStructureFactor"].name + ".mda"
    )
    settings["output_files"] = ["dummy_name", ["FileInMemory"], "no logs"]
    job_instance.setup(settings)
    job_instance.run(settings, status=True)
    results = job_instance.results
    main_name, axes_names = find_main_result(results)
    dependent_variables = {"SQw": results[main_name][:]}
    independent_variables = {name.split("/")[-1]: results[name][:] for name in axes_names}
    for axis_name in independent_variables:
        if "omega" in axis_name:
            independent_variables[axis_name] *= ENERGY_MDANSE_TO_MDMC
        elif "q" in axis_name:
            independent_variables[axis_name] *= Q_MDANSE_TO_MDMC
        elif "r" in axis_name:
            independent_variables[axis_name] *= R_MDANSE_TO_MDMC
        elif "time" in axis_name:
            independent_variables[axis_name] *= TIME_MDANSE_TO_MDMC
    errors = {"SQw": [np.sqrt(dependent_variables["SQw"][0])]}
    for file_object in output_temp_files.values():
        tmp_path = Path(file_object.name)
        tmp_path.unlink(missing_ok=True)
    return dependent_variables, independent_variables, errors


def get_default_mdanse_settings(job_name: str) -> dict[str, Any]:
    job_name = job_aliases.get(job_name, job_name)
    temp_instance = IJob.create(job_name, trajectory_input="mdmc")
    defaults = {}
    for key, value in temp_instance.settings.items():
        if "default" in value[1]:
            defaults[key] = value[1]["default"]
        else:
            temp_conf = IConfigurator.create(value[0], key)
            defaults[key] = temp_conf._default
    return defaults


def create_mdanse_resolution(
    ueV_resolution: float, resolution_function: str = "Gaussian", centre: float = 0.0
):
    if ueV_resolution is None or np.isclose(ueV_resolution, 0.0):
        return ("Ideal", {})
    ueV_to_radperps = measure(1.0, iunit="ueV", ounit="rad/ps", equivalent=True).toval()
    width_mdanse = ueV_resolution * ueV_to_radperps
    centre_mdanse = centre * ueV_to_radperps
    temp_instance = IInstrumentResolution.create(resolution_function)
    par_dict = {}
    par_dict.update(temp_instance.settings)
    for key in par_dict:
        if "sigma" in key:
            par_dict[key] = width_mdanse
        elif "mu" in key:
            par_dict[key] = centre_mdanse
    return (resolution_function, par_dict)


def create_mock_trajectory(time_step: float, n_steps: int, universe: Universe | None = None) -> str:
    input_data = {
        "parameters": {
            "box_repetitions": [1, 1, 1],
            "pbc": True,
            "time_step": time_step,
            "number_of_frames": n_steps,
        },
        "coordinates": [],
        "modulations": [],
    }
    if universe is not None:
        unit_cell = universe.dimensions
        input_data["parameters"]["box_size"] = list(list(line) for line in np.diag(unit_cell))
    return json.dumps(input_data)


def check_if_main(name: str, hdf5_node: h5py.Group | h5py.Dataset) -> tuple[str, list[str]] | None:
    if "tags" in hdf5_node.attrs:
        tags = hdf5_node.attrs["tags"].split(",")
        if "main" in tags and "partial" not in tags:
            main_result = hdf5_node.name
            axes = hdf5_node.attrs["axis"].split("|")
            return main_result, axes
    return None


def find_main_result(data_structure: h5py.File) -> tuple[str, list[str]]:
    return data_structure.visititems(check_if_main)


@ObservableFactory.register(("MDANSEObservable", "MDANSE"))
class MDANSEObservable(Observable):
    """Runs a specific MDANSE analysis on the input trajectory."""

    def __init__(self, mdanse_job_type: str, pick_dataset: str | None = None):
        super().__init__()
        self._name = "MDANSE"
        self.job_type = job_aliases.get(mdanse_job_type, mdanse_job_type)
        self.job_settings = {}
        self.job_instance = None
        self._independent_variables = None
        self._dependent_variables = None
        self._errors = None
        self._q_shells = []
        self._override_dataset = pick_dataset

    @property
    def independent_variables(self):
        return self._independent_variables

    @independent_variables.setter
    def independent_variables(self, input_dict):
        self._independent_variables = input_dict
        if "Q" in input_dict:
            target_q = input_dict["Q"]
            q_step = (np.max(target_q) - np.min(target_q)) / (max(len(target_q) - 1, 1))
            self._q_shells = [
                10 * float(np.min(target_q)),
                10 * float(np.max(target_q) + 0.1 * q_step),
                10 * float(q_step),
            ]

    @property
    def dependent_variables(self):
        return self._dependent_variables

    @dependent_variables.setter
    def dependent_variables(self, input_dict):
        self._dependent_variables = input_dict

    @property
    def errors(self):
        if self._errors is None:
            label = first(self.dependent_variables.keys())
            self._errors = {label: [np.sqrt(self.dependent_variables[label][0])]}
        return self._errors

    def minimum_frames(self, dt=None) -> int:
        return 0

    def maximum_frames(self) -> int:
        return 1e12

    def calculate_from_MD(self, MD_input, file_path: Path | None = None, verbose=0, **parameters):
        """Evaluate the function using the current parameter values.

        Gets the current values of parameters from trajectory attributes.

        Parameters
        ----------
        MD_input : CompactTrajectory
            An empty trajectory with a parameters attribute.
        verbose : int, optional
            Ignored, by default 0.
        """
        self._origin = "MD"
        if self.job_type in {"SQw", "NeutronDynamicTotalStructureFactor"}:
            print("Running multiple calculations for S(Q,w).")
            dep, indep, err = run_ndtsf_special_case(
                MD_input,
                file_path=file_path,
                verbose=verbose,
                q_shells=self._q_shells,
                **self.job_settings,
            )
            self._dependent_variables = dep
            self._independent_variables = indep
            self._errors = err
            return
        print(f"Running a single calculation of {self.job_type}.")
        self.job_instance, settings = self.build_job_instance(
            self.job_type,
            frames=len(MD_input),
            corr_frames=self.job_settings.get("correlation_frames", len(MD_input) // 2),
        )
        if self._q_shells is not None and "q_vectors" in settings:
            settings["q_vectors"][1]["shells"] = self._q_shells
        settings["trajectory"] = file_path
        self.job_instance.setup(settings)
        self.job_instance.run(settings, status=True)
        results = self.job_instance.results
        if self._override_dataset is None:
            main_name, axes_names = find_main_result(results)
        else:
            main_name = self._override_dataset
            axes_names = results[main_name].attrs["axis"].split("|")
        self._dependent_variables = {self.job_type: results[main_name][:]}
        self._independent_variables = {name.split("/")[-1]: results[name][:] for name in axes_names}
        self._errors = {self.job_type: [np.sqrt(self._dependent_variables[self.job_type][0])]}

    def initial_parameters(self):
        if self.job_type == "NeutronDynamicTotalStructureFactor":
            par_dict = get_default_mdanse_settings("DynamicIncoherentStructureFactor")
            par_dict.update(get_default_mdanse_settings("DynamicCoherentStructureFactor"))
            return par_dict
        return get_default_mdanse_settings(self.job_type)

    def set_parameters(self, input_dict: dict[str, Any]):
        self.job_settings.update(input_dict)
        if "q_vectors" in input_dict and "shells" in input_dict["q_vectors"][1]:
            self._q_shells = input_dict["q_vectors"][1]["shells"]

    def build_job_instance(
        self,
        job_type: str,
        frames: int = 1000,
        corr_frames: int = 500,
        frame_step: int = 1,
        trajectory_type: str = "mdanse",
    ):
        job_instance = IJob.create(job_type, trajectory_input=trajectory_type)
        settings = get_default_mdanse_settings(job_type)
        if "frames" in settings:
            settings["correlation_frames"] = corr_frames
            settings["frames"] = [0, frames, frame_step, settings["correlation_frames"]]
        settings["instrument_resolution"] = ("ideal", {})
        for key, value in self.job_settings.items():
            settings[key] = value
        settings["output_files"] = ["dummy_name", ["FileInMemory"], "no logs"]
        return job_instance, settings

    def predict_output(
        self,
        time_step: float = 1.0,
        total_frames: int = 1000,
        frame_step: int = 5,
        correlation_frames: int = 500,
        universe: Universe | None = None,
    ) -> list[tuple[str, npt.NDArray[np.floating], str]]:
        if self.job_type == "NeutronDynamicTotalStructureFactor":
            job_instance, job_settings = self.build_job_instance(
                "DynamicCoherentStructureFactor",
                frames=total_frames,
                corr_frames=correlation_frames,
                frame_step=frame_step,
                trajectory_type="mock",
            )
        else:
            job_instance, job_settings = self.build_job_instance(
                self.job_type,
                frames=total_frames,
                corr_frames=correlation_frames,
                frame_step=frame_step,
                trajectory_type="mock",
            )
        traj_string = create_mock_trajectory(
            time_step,
            total_frames,
            universe=universe,
        )
        job_settings["trajectory"] = traj_string
        if self._q_shells is not None and "q_vectors" in job_settings:
            job_settings["q_vectors"][1]["shells"] = self._q_shells
        job_instance.setup(job_settings)
        results = []
        for entry in job_instance.preview_output_axis():
            axis_name = entry[0].lower()
            axis_vals = np.array(entry[1])
            if "energy" in axis_name:
                axis_vals *= ENERGY_MDANSE_TO_MDMC
                unit = "meV"
            elif "q vector" in axis_name:
                axis_vals *= Q_MDANSE_TO_MDMC
                unit = "1/Ang"
            elif "distance" in axis_name:
                axis_vals *= R_MDANSE_TO_MDMC
                unit = "Ang"
            elif "time" in axis_name:
                axis_vals *= TIME_MDANSE_TO_MDMC
                unit = "fs"
            else:
                unit = "unknown unit"
            results.append((axis_name, summarise_array(axis_vals), unit))
        return results

    def read_from_file(self, reader: csv_reader):
        """Load the data from a file."""
        self._origin = "experiment"
        if not reader.parse_has_run:
            with reader:
                reader.parse()
        self._dependent_variables = reader.dependent_variables
        self._independent_variables = reader.independent_variables
        self._errors = reader.errors

    @property
    def uniformity_requirements(self):
        return None

    @property
    def dependent_variables_structure(self):
        return {first(self.dependent_variables.keys()): list(self.independent_variables.keys())}
