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

"""The Gaussian-Process-Regression minimizer class"""

from __future__ import annotations

import itertools
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.ndimage import minimum, minimum_position
from sklearn.gaussian_process import GaussianProcessRegressor as skGPR
from sklearn.gaussian_process import kernels

from MDMC.MD.parameters import Parameter, Parameters
from MDMC.refinement.minimizers.minimizer_abs import Minimizer

if TYPE_CHECKING:
    from MDMC.control import Control


class GPR(Minimizer):
    """
    ``Minimizer`` employing Gaussian Process Regression. Creates a predefined array of points
    across all parameters and performs a simulation at each point. Then performs a Gaussian
    process regression fit across all measured points and predicts across a finer grid, before
    returning the predicted minimum FoM and the associated parameter values. This minimizer
    works best when physically realistic constraints are applied, in the absence of being
    provided with them, the minimizer sets constraints equal to 20% of the current parameter
    values.

    Parameters
    ----------
    control: Control
        The ``Control`` object which uses this Minimizer.
    parameters: Parameters
        The parameters in the simulation Universe to be optimized

    Attributes
    ----------
    history_columns: list[str]
        list of the column titles, and parameter names in the minimizer history
    """

    def __init__(
        self,
        control: Control,
        parameters: Parameters,
        previous_history: Path | str | None = None,
        **settings: Any,
    ):
        super().__init__(control, parameters, previous_history)
        np.random.seed(0)  # This should mean results are reproducible in tests

        (self.parameter_names, self.parameter_point_array) = self.create_parameter_point_array(
            parameters,
        )
        self.results_filename = settings.get("results_filename")
        self.change_parameters()
        self.previous_history = previous_history
        self.state_changed = False

    def create_parameter_point_array(self, parameters: Parameters) -> tuple[list[str], list[tuple]]:
        """
        Takes or creates the constraints of the parameters to be minimised and makes an array
        of points, placed on a Latin hypercube covering the space defined by the constraints.
        The resulting array of coordinates is self.control.n_steps long.

        Parameters
        ----------
        parameters : Parameters
            All ``Parameter`` objects that are being refined.

        Returns
        -------
        parameter_names : list
                Ordered ``list`` of names of parameters
        point_array : list
                ``list`` of parameter coordinates to be simulated
        """
        parameter_names = [str(name) for name in parameters]

        samples = st.qmc.LatinHypercube(d=len(parameters), scramble=False, seed=1)
        try:
            latin_points = samples.random(n=self.control.n_steps)
        except AttributeError as error:
            raise AttributeError(
                "GPR requires that the number of refinement steps "
                "is set when initialising Control.",
            ) from error

        lower_bounds = [self.create_bounds(parameter)[0] for parameter in parameters.values()]
        upper_bounds = [self.create_bounds(parameter)[1] for parameter in parameters.values()]

        latin_points = st.qmc.scale(latin_points, lower_bounds, upper_bounds)
        return parameter_names, latin_points

    @staticmethod
    def create_bounds(parameter: Parameter, fraction: float = 0.3) -> tuple[float, float]:
        """
        Returns either the parameter constraints (bounds) or bounds for each parameter
        equal to the parameter value =/- fraction*parameter.value, defaulting to +-30%.
        Raises a ValueError if value is zero and has no constraints since it is not possible
        to make a guess.

        Parameters
        ----------
        parameter : Parameter
            A MDMC ``Parameter``
        fraction : optional, float
            The fractional size of the bound, defaults to 0.3 == +-30%

        Returns
        -------
        lower_bound : float
            The lower bound for the parameter
        upper_bound : float
            The upper bound for the parameter

        Raises
        -----
        ValueError
            If `parameter.value` is zero and no constraints have been set for it there
            is no sensible way to guess bounds.
        """
        try:
            lower_bound = parameter.constraints[0]
            upper_bound = parameter.constraints[1]
        except TypeError as error:
            if parameter.value > 0:
                lower_bound = parameter.value * (1.0 - fraction)
                upper_bound = parameter.value * (1.0 + fraction)
            elif parameter.value < 0:
                lower_bound = parameter.value * (1.0 + fraction)
                upper_bound = parameter.value * (1.0 - fraction)
            else:
                raise ValueError(
                    f"You have set parameter {parameter.name} value to zero and "
                    "have no constraints set for it. "
                    "Please set constraints for it",
                ) from error

        parameter.constraints = [lower_bound, upper_bound]

        return lower_bound, upper_bound

    def has_converged(self) -> bool:
        """
        Checks if the refinement process has finished, i.e. if all points across
        the parameter_point_array have been measured.

        Returns
        -------
        bool
            Whether or not the minimizer has converged.
        """
        return len(self.history) >= self.control.n_steps

    @property
    def history_columns(self) -> list[str]:
        """
        Returns column labels of the history

        Returns
        -------
        list[str]
            A ``list`` of ``str`` containing all the column labels in the history
        """
        return ["FoM"] + list(self.parameters)

    def set_parameter_values(self, parameter_names: list[str], values: list[float]) -> None:
        """
        Assigns a new value to each parameter (specified by the parameter.name)

        Parameters
        ----------
        parameter_names : list[str]
            A list of the names of the parameters whose values are to be set
        values : list[float]
            A list of the values to be set for each parameter
        """

        for name, value in zip(parameter_names, values):
            self.parameters[name].value = value

    def change_parameters(self) -> None:
        """
        Selects a new value for each parameter from the array of parameter values to interrogate
        from the parameter_point_array.
        """

        point_to_calculate = len(self._history)
        if point_to_calculate <= len(self.parameter_point_array):
            coordinates = self.parameter_point_array[point_to_calculate]
            self.set_parameter_values(self.parameter_names, coordinates)

    def step(self, FoM: float) -> None:
        """
        Increments the minimization by a step

        Parameters
        ----------
        FoM : float
            The current figure of merit value.
        """

        self.FoM = FoM
        history = [self.FoM]
        self.FoM_old = self.FoM
        self.state_changed = True

        values = np.array([self.parameters[p].value for p in self.parameters])
        history.extend(values)
        self._history.append(history)
        if not self.has_converged():
            self.change_parameters()

    def reset_parameters(self) -> None:
        """Resets the Parameter values to the last set of values in parameter_point_array"""

        for i, parameter in enumerate(self.parameters):
            self.parameters[parameter].value = self.parameter_point_array[-1][i]

    def GPR_fit(
        self,
        filename: str | None = None,
        alpha: float | None = 5,
        length_scale: float | None = 4,
    ):
        """
        Reads in the contents of the supplied filename, assumes it is the output of a refinement
        and can be read into a dataframe with the relevant parameters. Uses the recorded points
        file to perform a Gaussian process regression
        (https://scikit-learn.org/stable/modules/gaussian_process.html) and fit the points
        to some kernel, here using an RBF kernel.

        Parameters
        ----------
        filename: str, optional
            The filename or full path to a comma separated value file containing
            the full output of the refinement. Defaults to None, but if results_filename is set by
            Control then this is passed into GPR as self.results_filename and used here.
        alpha: float, optional
            Hyperparameter for the fitting, which can represent Gaussian noise in measurement
            points, e.g. how much variation in the output you expect between MD runs.
            Defaults to 5.
        length_scale: float, optional
            Hyperparameter for the fitting, which can represent how quickly the kernel is able
            to change/oscillate. Defaults to 4.0.

        Returns
        -------
        GaussianProcessRegressor : GaussianProcessRegressor
            The fitted points using GPR

        Minimum figure of merit : float

        Minimum parameter values : float
        """

        if not filename:
            filename = self.results_filename

        records = pd.read_csv(filename, delimiter=",")
        records = records.astype(dtype=float, errors="ignore")
        # Convert to float where possible (i.e. not a string)

        FOMs = records["FoM"].to_list()
        min_FOM = np.min(FOMs)
        min_par_index = records.index[records["FoM"] == min_FOM].tolist()

        records = records.drop(columns=["Unnamed: 0", "FoM", "Change state"], errors="ignore")
        # TODO this is hard coded to creation of history, may want to change

        min_pars = records.loc[min_par_index]

        coordinates = records.values.tolist()

        kernel = kernels.RBF(length_scale=np.ones(len(coordinates[0])) * length_scale)
        gpr = skGPR(kernel, n_restarts_optimizer=50, alpha=alpha)

        fitted_GPR = gpr.fit(coordinates, FOMs)

        return fitted_GPR, min_FOM, min_pars

    @staticmethod
    def GPR_predict(
        input_regressor,
        points: float | None = 100,
    ) -> tuple[list[tuple[float]], np.ndarray]:
        """
        Takes a fitted Gaussian process regressor from GPR_fit, creates an fine array of points
        between the minimum and maximum measured parameter values and predicts the FoM at each
        one of these points.

        Parameters
        ----------
        input_regressor : GaussianProcessRegressor
            A fitted Gaussian Process regressor object
        points: int, optional
            Number of points to predict the GPR over. Defaults to 100

        Returns
        -------
        point_array : list[tuple[float]]
            The ``list`` of coordinates at which the predictions are made
        prediction : numpy.ndarray
            A ``list`` of predicted figure of merit surface at each coordinate in the point_array
        """

        regressor_points = input_regressor.X_train_

        predictive_coordinates = []

        for column in regressor_points.T:
            min_point, max_point = np.min(column), np.max(column)
            dense_array = np.linspace(min_point, max_point, points)
            predictive_coordinates.append(dense_array)

        point_array = list(itertools.product(*predictive_coordinates))
        # predict method needs explicit array
        prediction = input_regressor.predict(point_array, return_std=False)

        return point_array, prediction

    @staticmethod
    def global_minimum_position(
        predicted_FOMs: np.ndarray,
        measured_parameter_coordinates: list[float],
    ) -> tuple[np.ndarray, float]:
        """
        Gives the coordinates of the global minimum of the predicted figure of merit surface.

        Parameters
        ----------
        predicted_FOMs : numpy.ndarray
            A numpy ``array`` of the predicted figures of merit
        measured_parameter_coordinates: list
            A ``list`` of the coordinates corresponding to the points at which the FoM was predicted

        Returns
        -------
        minimum_parameters : numpy.ndarray
            The parameter coordinates where the minimum figure of merit is predicted to be
        min_FoM : float
            The predicted minimum figure of merit value
        """

        min_coordinates = minimum_position(predicted_FOMs)[0]
        min_FoM = minimum(predicted_FOMs)
        minimum_parameters = measured_parameter_coordinates[min_coordinates]

        return minimum_parameters, min_FoM

    def extract_result(self) -> list:
        """
        Extracts the measured & predicted FoM and point(s)

        Returns
        -------
        list
            A list of: coordinates of lowest FoM, Minimum FoM, Coordinate of best predicted FoM,
            Minimum predicted FoM
        """
        fit, min_FoM_measured, min_parameters_measured = self.GPR_fit()
        points, FoMs = self.GPR_predict(fit)
        min_parameters_predicted, min_FoM_predicted = self.global_minimum_position(FoMs, points)
        self.set_parameter_values(self.parameter_names, min_parameters_predicted)

        min_parameters_measured = tuple(min_parameters_measured.iloc[0])

        return [
            min_parameters_measured,
            min_FoM_measured,
            min_parameters_predicted,
            min_FoM_predicted,
        ]

    def format_result_string(self, minimizer_output: list) -> str:
        """
        Parameters
        ----------
        minimizer_output: list
            A list of: coordinates of lowest FoM, Minimum FoM, Coordinate of best predicted FoM,
            Minimum predicted FoM

        Returns
        -------
        str
            An output string, formatted with the appropriate information about measured
            and predicted points
        """

        finished = "finished" if self.has_converged() else "not finished"
        converged_message = f"The refinement has {finished}."

        # as of numpy 2.0.0, np.float64 has repr e.g. "np.float64(3.14)" instead of "3.14"
        # we use legacy print options to make the string nicer with less fiddling
        with np.printoptions(legacy="1.25"):
            output_string = f"""
                            {converged_message}

                            Minimum measured point is:
                            {minimizer_output[0]} with an
                            FoM of {minimizer_output[1]}.

                            Minimum point predicted is:
                            {minimizer_output[2]} for an
                            FoM of {minimizer_output[3]}.
                            """

            return dedent(output_string)
