"""
Module for AbstractSQw and total SQw class.
"""

import numpy as np
from numpy.testing import assert_allclose

from MDMC.common import units
from MDMC.common.constants import h, h_bar
from MDMC.common.decorators import unit_decorator_getter
from MDMC.resolution.resolution_factory import ResolutionFactory
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory
from MDMC.trajectory_analysis.observables.obs import Observable
from MDMC.trajectory_analysis.observables.obs_factory import ObservableFactory
from MDMC.utilities.trajectory_slicing import slice_trajectory


def gaussian_2D(
    x_axis,
    y_axis,
    centre_x: float = 5.0,
    centre_y: float = 5.0,
    width_x: float = 0.5,
    width_y: float = 0.5,
) -> np.ndarray:
    x_results = (np.sqrt(2.0 * np.pi) / width_x) * np.exp(
        -0.5 * ((x_axis - centre_x) / width_x) ** 2,
    )
    y_results = (np.sqrt(2.0 * np.pi) / width_y) * np.exp(
        -0.5 * ((y_axis - centre_y) / width_y) ** 2,
    )
    return y_results.reshape((len(y_results), 1)) * x_results.reshape((1, len(x_results)))

@ObservableFactory.register(('FakeObservable', 'gauss2D'))
class FakeObservable(Observable):
    def __init__(self):
        super().__init__()
        self._name = "gauss2D"
        self.x_axis = np.linspace(-5.0, 25.0, 111)
        self.y_axis = np.linspace(-6.0, 33.0, 210)
        self._independent_variables = None
        self._dependent_variables = None
        self._errors = None

    @property
    def independent_variables(self):
        if self._independent_variables is None:
            self._independent_variables = {"x": self.x_axis, "y": self.y_axis}
        return self._independent_variables

    @independent_variables.setter
    def independent_variables(self, input_dict):
        self._independent_variables = input_dict

    @property
    def dependent_variables(self):
        if self._dependent_variables is None:
            self._dependent_variables = {"gauss2D": gaussian_2D(self.x_axis, self.y_axis)}
        return self._dependent_variables

    @dependent_variables.setter
    def dependent_variables(self, input_dict):
        self._dependent_variables = input_dict

    @property
    def errors(self):
        if self._errors is None:
            self._errors = {"gauss2D": np.sqrt(self.dependent_variables["gauss2D"])}
        return self._errors

    def minimum_frames(self, dt=None) -> int:
        return 0

    def maximum_frames(self) -> int:
        return 1e12

    def calculate_from_MD(self, MD_input, verbose=0, **parameters):
        self._origin = 'MD'
        self._dependent_variables = {"gauss2D": gaussian_2D(self.x_axis,
                           self.y_axis,
                           centre_x=MD_input.parameters.get("centre_x"),
                           centre_y=MD_input.parameters.get("centre_y"),
                           width_x=MD_input.parameters.get("width_x"),
                           width_y=MD_input.parameters.get("width_y"))}
        self._errors = {"gauss2D": np.sqrt(self._dependent_variables["gauss2D"])}

    def read_from_file(self, reader, file_name):
        self._origin = 'experiment'
        self._dependent_variables = {"gauss2D": gaussian_2D(self.x_axis,
                           self.y_axis,
                           centre_x=5.0,
                           centre_y=4.0,
                           width_x=3.3,
                           width_y=2.1)}
        self._errors = {"gauss2D": np.sqrt(self._dependent_variables["gauss2D"])}

    @property
    def uniformity_requirements(self):
        return None

    @property
    def dependent_variables_structure(self):
        return {'gauss2D': ['x', 'y']}
