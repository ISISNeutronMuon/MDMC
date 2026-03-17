"""
Wrapper for analysis run using MDANSE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
from MDANSE.Framework.Configurators.IConfigurator import IConfigurator
from MDANSE.Framework.Jobs.IJob import IJob
from more_itertools import first

from MDMC.trajectory_analysis.observables.obs import Observable
from MDMC.trajectory_analysis.observables.obs_factory import ObservableFactory

if TYPE_CHECKING:
    from MDMC.readers.observables.csv_reader import csv_reader


def get_default_mdanse_settings(job_name: str) -> dict[str, Any]:
    temp_instance = IJob.create(job_name, trajectory_input="mdmc")
    defaults = {}
    for key, value in temp_instance.settings.items():
        if "default" in value[1]:
            defaults[key] = value[1]["default"]
        else:
            temp_conf = IConfigurator.create(value[0], key)
            defaults[key] = temp_conf._default
    return defaults


def check_if_main(hdf5_node: h5py.Group | h5py.Dataset) -> tuple[str, list[str]] | None:
    if "tags" in hdf5_node.attrs and "main" in hdf5_node.attrs["tags"].split(","):
        main_result = hdf5_node.name
        axes = hdf5_node.attrs["axis"].split("|")
        return main_result, axes
    return None


def find_main_result(data_structure: h5py.File) -> tuple[str, list[str]]:
    return data_structure.visit(check_if_main)


@ObservableFactory.register(("MDANSEObservable", "MDANSE"))
class MDANSEObservable(Observable):
    """Runs a specific MDANSE analysis on the input trajectory."""

    def __init__(self, mdanse_job_type: str):
        super().__init__()
        self._name = "MDANSE"
        self.job_type = mdanse_job_type
        self.job_settings = {}
        self._independent_variables = None
        self._dependent_variables = None
        self._errors = None

    @property
    def independent_variables(self):
        return self._independent_variables

    @independent_variables.setter
    def independent_variables(self, input_dict):
        self._independent_variables = input_dict

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

    def calculate_from_MD(self, MD_input, verbose=0, **parameters):
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
        self.job_instance = IJob.create(self.job_type, trajectory_input="mdmc")
        settings = get_default_mdanse_settings(self.job_type)
        settings["frames"] = [0, len(MD_input), 1, len(MD_input) // 2]
        for key, value in self.job_settings.items():
            settings[key] = value
        settings["trajectory"] = MD_input
        settings["output_files"] = ["dummy_name", ["FileInMemory"], "no logs"]
        self.job_instance.setup(settings)
        self.job_instance.run(settings, status=True)
        results = self.job_instance.results
        main_name, axes_names = find_main_result(results)
        self._dependent_variables = {self.job_type: [results[main_name]]}
        self._independent_variables = {name.split("/")[-1]: results[name] for name in axes_names}
        self._errors = {self.job_type: [np.sqrt(self._dependent_variables[self.job_type][0])]}

    def read_from_file(self, reader: csv_reader):
        """Load the data from a file."""
        self._origin = "experiment"
        self._dependent_variables = reader.dependent_variables
        self._independent_variables = reader.independent_variables
        self._errors = reader.errors
        self._units = reader.units

    @property
    def uniformity_requirements(self):
        return None

    @property
    def dependent_variables_structure(self):
        return {first(self.dependent_variables.keys()): list(self.independent_variables.keys())}
