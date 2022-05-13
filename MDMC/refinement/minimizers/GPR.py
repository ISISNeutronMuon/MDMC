"""The Gaussian-Process-Regression minimizer class"""
import itertools

import numpy as np
import scipy.stats as st
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor as skGPR
from sklearn.gaussian_process import kernels
from scipy.ndimage import minimum_position, minimum

from MDMC.refinement.minimizers.minimizer_abs import Minimizer


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
    hypercube : optional, bool
        Boolian toggle for is the n_points should be placed in a latin hypercube, or as a grid
        across each parameter. Defaults to False
    alpha : float
        Hyperparameter for the fitting, also can represent additional Gaussian noise in measurement
        points
    length_scale : float
        Hyperparameter for the fitting, a lengthscale parameter for the kernel.
    n_points : int
        A number of points which will be measured at, either randomly on a latin hypercube
        (if hypercube=True) or p^n_points (p = number of parameters) in a regular grid
        (if hypercube=False)


    Attributes
    ----------
    history_columns: list[str]
        list of the column titles for the minimizer history
    """


    def __init__(self, parameters, distribution, max_parameter_change, **settings):
        super().__init__(parameters, distribution, max_parameter_change)
        self.hypercube = settings.get('hypercube', False)
        self.alpha = settings.get('alpha', 0.01)
        self.length_scale = settings.get('length_scale', 0.1)
        n_points = settings.get('n_points', 4)

        self.parameter_names, self.parameter_point_array = \
        self.create_parameter_point_array(parameters, n_points)

        self.change_parameters()


    def create_parameter_point_array(self, parameters, points=2):
        """
        Takes or creates the constraints of the parameters to be minimised, the either makes an
        array of length "points" and performs the Cartesian product across all parameters, to
        give an equally spaced set of coordinates to be measured. This set of coordinates will
        be points^(#of dimensions long). If self.hypercube is True then an array of points will
        be placed on a Latin hypercube covering the space defined by the constraints. The resulting
        array of coordinates will be "points" long.

        Parameters
        ----------
        parameters : list
            All ``Parameter`` objects that are being refined.
        points : int, optional
            Number of points across the constraint range to take, defaults to 2.

        Returns
        -------
        parameter_names : list
                Ordered list of names of parameters
        point_array : array
                Array of parameter coordinates to be simulated
        """
        parameter_names = []
        bounds_array = []
        point_array = []
        lower_bounds = []
        upper_bounds = []

        if self.hypercube:
            samples = st.qmc.LatinHypercube(d=len(parameters), centered=True)
            latin_points = samples.random(n=points)

            for name, parameter in parameters.items():
                lower_bound, upper_bound = self.create_bounds(parameter)
                parameter_names.append(str(name))
                lower_bounds.append(lower_bound)
                upper_bounds.append(upper_bound)

            latin_points = st.qmc.scale(latin_points, lower_bounds, upper_bounds)
            return parameter_names, latin_points

        for name, parameter in parameters.items():
            lower_bound, upper_bound = self.create_bounds(parameter)
            parameter_names.append(str(name))

            bounds_grid = np.linspace(lower_bound, upper_bound, points)
            bounds_array.append(bounds_grid)
        point_array =  list(itertools.product(*bounds_array))
        # * is necessary for unpacking the arrays
        return parameter_names, point_array


    def create_bounds(self, parameter, fraction = 0.2):
        """
        Returns either the parameter constraints (bounds) or some sensible bounds for
        a given parameter, defaults to +-20%. Raises a ValueError if value is zero and has
        no constraints as no educated guess is possible.

        Parameters
        ----------
        parameter : Parameter instance
            A instance of the MDMC Parameter class
        fraction : optional, float
            The fractional size of the bound, defaults to 0.2 == +-20%

        Returns
        -------
        lower_bound : float
            The lower bound for the parameter
        upper_bound : float
            The upper bound for the parameter

        Raises
        -----
        ValueError
            If parameter.value is zero and no constraints have been set for it there
            is no sensible way to guess bounds.
        """
        try:
            lower_bound = parameter.constraints[0]
            upper_bound = parameter.constraints[1]
        except TypeError:
            if not parameter.value ==0:
                lower_bound = parameter.value*(1.0 - fraction)
                upper_bound = parameter.value*(1.0 + fraction)
            else:  # pylint: disable=raise-missing-from
                raise ValueError(f'You have set parameter {parameter.name} value to zero and \
                    have no constraints set for it. Please set constraints for it')
        return lower_bound, upper_bound

    def has_converged(self, conv_tol: float = 1e-5, min_steps: int =1) -> bool:
        """
        Checks if the refinement process has converged on a stable solution.
        Specifically, it checks if the certainty of the parameters being refined have all
        become less than the relative conversion tolerance (`conv_tol`).
        It also allows specifying a minimum
        number of refinement steps (`min_steps`) that must have been simulated
        before checking for convergence.

        Parameters
        ----------
        conv_tol : float, optional
            The relative tolerance of the convergence check. Defaults to `1e-5`
        min_steps : int, optional
            The number of refinement steps after which
            convergence is checked. If the number of accepted state changes is less than this,
            then the refinement is deemed as not converged.
            Defaults to `1`.

        Returns
        -------
        bool
            Whether or not the minimizer has converged.
        """
        converged = False
        run_steps = np.max([min_steps, len(self.parameter_point_array)])

        if len(self.history) >= run_steps:
            converged = True

        return converged

    def change_state(self):
        #change_state = self.comm.bcast(change_state, root=0)
        return True

    @property
    def history_columns(self):

        return ['FoM', 'Change state'] + [list(self.parameters)]

    # pylint: disable=arguments-differ
    # we allow implementations of the abstract method to have different arguments

    def set_parameter_values(self, parameter_names, values):
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
            self.parameters[str(name)].value = value

    def change_parameters(self, parameters=None):
        """
        Selects a new value for each parameter from the array of parameter values to interrogate

        Parameters
        ----------
        parameters : list
            All ``Parameter`` objects that are being refined
        """

        point_to_calculate = len(self._history)
        if point_to_calculate <= len(self.parameter_point_array):
            coordinates = self.parameter_point_array[point_to_calculate]
            self.set_parameter_values(self.parameter_names, coordinates)

    def step(self, FoM):
        """
        Increments the minimization by a step
        """

        self.FoM = FoM
        values = np.array([self.parameters[str(p)].value for p in self.parameters])
        history = [self.FoM]
        history.append('Accepted')
        self.FoM_old = self.FoM
        self.state_changed = True

        history.extend(values)
        self._history.append(history)
        if len(self._history) < len(self.parameter_point_array):
            self.change_parameters()

    def reset_parameters(self):
        """
        Resets the ``Parameter`` values to the first step
        """
        for i, parameter_name in enumerate(self.parameter_names):
            for parameter in self.parameters:
                if parameter.name == parameter_name:
                    parameter.value = self.parameter_point_array[0][i]


    def GPR_fit(self, filename="results.csv", alpha=0.1):
        """
        Uses the measured points in the supplied file to perform a Gaussian
        process regression and fit the points.

        Parameters
        ----------
        filename : str, optional
            The filename or full path to a comma separated value file containing
            the full output of the refinement. Defaults to the results.csv
            produced by the refinement.
        alpha: float, optional
            The intrinsic uncertainty associated with the measured points
            i.e. not assuming the measured point must lie on the underlying
            surface, but may fluctuate around it by some value

        Returns
        -------
        GaussianProcessRegressor
            The fitted points using GPR
        """
        records = pd.read_csv(filename, delimiter=',')
        records = records.astype(dtype=float, errors='ignore')
        # Convert to float where possible (i.e. not a string)

        FOMs = records['FoM'].to_list()

        records = records.drop(columns=['Unnamed: 0', 'FoM', 'Change state'])
        # TODO this is hard coded to creation of history, may want to change
        coordinates = records.values.tolist()

        kernel = kernels.RBF(length_scale = np.ones(len(coordinates[0]))*self.length_scale)
        gpr = skGPR(kernel, n_restarts_optimizer=50, alpha = alpha)

        fitted_GPR = gpr.fit(coordinates, FOMs)

        return fitted_GPR

    def GPR_predict(self, input_regressor, points=100):
        """
        Takes a fitted Gaussian process regressor, creates an array of points between the
        minimum and maximum measured parameter values and predicts the FoM on these points

        Parameters
        ----------
        input_regressor : GaussianProcessRegressor object
            A fitted Gaussian Process regressor object
        points: int, optional
            Number of points to predict the GPR over. Defaults to 100

        Returns
        -------
        point_array : list
            The list of coordinates at which the predictions are made
        prediction : array
            Array of predicted figure of merit surface at each coordinate in the point_array
        """
        regressor_points = input_regressor.X_train_

        predictive_coordinates = []

        for column in regressor_points.T:
            min_point, max_point = np.min(column), np.max(column)
            dense_array = np.linspace(min_point, max_point, points)
            predictive_coordinates.append(dense_array)

        point_array =  list(itertools.product(*predictive_coordinates))
        # predict method needs explicit array
        prediction = input_regressor.predict(point_array, return_std=False)

        return point_array, prediction

    def global_minimum_position(self, predicted_FOMs, measured_parameter_coordinates):
        """
        Gives the coordinates of the global minimum of the predicted figure of merit surface.

        Parameters
        ----------
        predicted_FOMs : array
            An array of the predicted figures of merit
        measured_parameter_coordinates: list
            A list of the coordinates corresponding to the points at which the FoM was predicted

        Returns
        -------
        minimum_parameters : array
            The parameter coordinates where the minimum figure of merit is predicted to be
        min_FoM : float
            The predicted minimum figure of merit value
        """

        min_coordinates = minimum_position(predicted_FOMs)[0]
        min_FoM = minimum(predicted_FOMs)
        minimum_parameters = measured_parameter_coordinates[min_coordinates]

        return minimum_parameters, min_FoM

    def present_result(self):
        """
        Sets the parameters those predicted to return the minimum FoM, returns
        the coordinates of the minima and the predicted FoM.

        Returns
        -------
        minima_coordinate : array(float)
            The parameter coordinates where the minimum figure of merit is predicted to be
        min_FoM : float
            The predicted minimum figure of merit value
        """
        fit = self.GPR_fit()
        points, FoMs = self.GPR_predict(fit)
        minima_coordinate, min_FoM = self.global_minimum_position(FoMs, points)
        self.set_parameter_values(self.parameter_names, minima_coordinate)

        return minima_coordinate, min_FoM
