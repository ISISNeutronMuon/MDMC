"""The Gaussian-Process-Regression minimizer class"""
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.optimize import direct
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

    def __init__(self, control: 'Control', parameters: Parameters, \
        previous_history: Optional[Union[Path, str]] = None,**settings: dict):

        super().__init__(control, parameters, previous_history)
        np.random.seed(0) # This should mean results are reproducible in tests

        self.parameter_names, self.parameter_point_array = \
        self.create_parameter_point_array(parameters)
        self.results_filename = settings.get('results_filename')
        self.change_parameters()
        self.previous_history = previous_history
        self.state_changed = False

    def create_parameter_point_array(self,
                                     parameters: Parameters) -> 'tuple[list[str], list[tuple]]':
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
            raise AttributeError("GPR requires that the number of refinement steps "
                                 "is set when initialising Control.") from error

        self.lower_bounds = [self.create_bounds(parameter)[0] for parameter in parameters.values()]
        self.upper_bounds = [self.create_bounds(parameter)[1] for parameter in parameters.values()]

        latin_points = st.qmc.scale(latin_points, self.lower_bounds, self.upper_bounds)
        return parameter_names, latin_points

    @staticmethod
    def create_bounds(parameter: Parameter, fraction: float = 0.3) -> 'tuple[float, float]':
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
                lower_bound = parameter.value*(1.0 - fraction)
                upper_bound = parameter.value*(1.0 + fraction)
            elif parameter.value < 0:
                lower_bound = parameter.value*(1.0 + fraction)
                upper_bound = parameter.value*(1.0 - fraction)
            else:
                raise ValueError(f'You have set parameter {parameter.name} value to zero and \
                    have no constraints set for it. Please set constraints for it') from error

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
    def history_columns(self) -> 'list[str]':
        """
        Returns column labels of the history

        Returns
        -------
        list[str]
            A ``list`` of ``str`` containing all the column labels in the history
        """
        return ['FoM'] + list(self.parameters)

    def set_parameter_values(self, parameter_names: 'list[str]', values: 'list[float]') -> None:
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

    def GPR_fit(self, filename: Optional[str] = None):
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

        Returns
        -------
        GaussianProcessRegressor : GaussianProcessRegressor
            The fitted points using GPR

        Minimum figure of merit : float

        Minimum parameter values : float
        """

        if not filename:
            filename = self.results_filename

        records = pd.read_csv(filename, delimiter=',')
        records = records.astype(dtype=float, errors='ignore')
        # Convert to float where possible (i.e. not a string)

        FOMs = records['FoM'].to_list()
        min_FOM = np.min(FOMs)
        min_par_index = records.index[records['FoM']==min_FOM].tolist()

        records = records.drop(columns=['Unnamed: 0', 'FoM', 'Change state'], errors='ignore')
        # TODO this is hard coded to creation of history, may want to change

        min_pars = records.loc[min_par_index]

        coordinates = records.values.tolist()

        kernel = kernels.Matern(length_scale = np.ones(len(coordinates[0]))) + kernels.WhiteKernel()
        gpr = skGPR(kernel, n_restarts_optimizer=50, alpha = 0.)

        fitted_GPR = gpr.fit(coordinates, FOMs)

        return fitted_GPR, min_FOM, min_pars

    def GPR_predict(self, input_regressor) -> 'tuple[list[tuple[float]], np.ndarray]':
        """
        Takes a fitted Gaussian process regressor from GPR_fit, and uses the DIRECT
        global optimization algorithm
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.direct.html#id1)
        to find the minimum FoM according to the regressor's predictions.

        Parameters
        ----------
        input_regressor : GaussianProcessRegressor
            A fitted Gaussian Process regressor object
        points: int, optional
            Number of points to predict the GPR over. Defaults to 100

        Returns
        -------
        min_params : numpy.ndarray
            The ``list`` of coordinates at which the best FoM was found
        min_fom : float
            The best FoM found by the optimizer
        """
        def predict_wrapper(x): return input_regressor.predict([x])

        bounds = list(zip(self.lower_bounds, self.upper_bounds))

        optimizer_result = direct(predict_wrapper, bounds)

        min_params = optimizer_result.x
        min_fom = optimizer_result.fun

        return min_params, min_fom

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
        min_parameters_predicted, min_FoM_predicted = self.GPR_predict(fit)
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

        if self.has_converged():
            converged_message = 'The refinement has finished.'
        else:
            converged_message = "The refinement has not finished."

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
