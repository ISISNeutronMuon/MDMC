"""
Wrapper for analysis run using MDANSE.
"""

from typing import Any

import h5py
import numpy as np
from MDANSE.Framework.Configurators.IConfigurator import IConfigurator
from MDANSE.Framework.Jobs.IJob import IJob

from MDMC.trajectory_analysis.observables.obs import Observable
from MDMC.trajectory_analysis.observables.obs_factory import ObservableFactory


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
    if "tags" in hdf5_node.attrs and "main" in hdf5_node.attrs["tags"].split(','):
        main_result = hdf5_node.name
        axes = hdf5_node.attrs["axis"].split("|")
        return main_result, axes
    return None


def find_main_result(data_structure: h5py.File) -> tuple[str, list[str]]:
    return data_structure.visit(check_if_main)


@ObservableFactory.register(('MDANSEObservable', 'MDANSE'))
class MDANSEObservable(Observable):
    """Runs a specific MDANSE analysis on the input trajectory.
    """
    def __init__(self):
        super().__init__()
        self._name = "MDANSE"
        self.job_type = "DensityOfStates"
        self.x_axis = np.linspace(-5.0, 25.0, 111)
        self.y_axis = np.linspace(-6.0, 33.0, 210)
        self._independent_variables = None
        self._dependent_variables = None
        self._errors = None

    @property
    def independent_variables(self):
        if self._independent_variables is None:
            self._independent_variables = {"y": self.y_axis, "x": self.x_axis}
        return self._independent_variables

    @independent_variables.setter
    def independent_variables(self, input_dict):
        self._independent_variables = input_dict

    @property
    def dependent_variables(self):
        if self._dependent_variables is None:
            self._dependent_variables = {"gauss2D": [gaussian_2D(self.x_axis, self.y_axis)]}
        return self._dependent_variables

    @dependent_variables.setter
    def dependent_variables(self, input_dict):
        self._dependent_variables = input_dict

    @property
    def errors(self):
        if self._errors is None:
            self._errors = {"gauss2D": [np.sqrt(self.dependent_variables["gauss2D"][0])]}
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
        self._origin = 'MD'
        self.job_instance = IJob.create(self.job_type, trajectory_input="mdmc")
        settings = get_default_mdanse_settings(self.job_type)
        for key, value in self.job_settings.items():
            settings[key] = value
        settings["trajectory"] = MD_input
        settings["output_files"] = ["dummy_name", ["FileInMemory"], "no logs"]
        self.job_instance.run(settings, status=True)
        results = self.job_instance.results
        main_name, axes_names = find_main_result(results)
        self._dependent_variables = {self.job_type: [results[main_name]]}
        self._independent_variables = {name.split("/")[-1]: results[name] for name in axes_names}
        self._errors = {self.job_type: [np.sqrt(self._dependent_variables[self.job_type][0])]}

    def read_from_file(self, reader, file_name):
        """Generate the target values from hardcoded arguments.

        The optimisation should produce the following arguments:
            centre_x=5.0
            centre_y=4.0
            width_x=3.3
            width_y=2.1
        """
        self._origin = 'experiment'
        self._dependent_variables = {"gauss2D": [gaussian_2D(self.x_axis,
                           self.y_axis,
                           centre_x=5.0,
                           centre_y=4.0,
                           width_x=3.3,
                           width_y=2.1)]}
        self._errors = {"gauss2D": [np.sqrt(self._dependent_variables["gauss2D"][0])]}

    @property
    def uniformity_requirements(self):
        return None

    @property
    def dependent_variables_structure(self):
        return {'gauss2D': ['y', 'x']}
