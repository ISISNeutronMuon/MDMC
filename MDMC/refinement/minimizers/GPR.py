"""The Gaussian-Process-Regression minimizer class"""
import numpy as np
import itertools
import scipy.stats as st
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process import kernels
from scipy.ndimage import minimum_position

from MDMC.refinement.minimizers.minimizer_abs import Minimizer


class GPR(Minimizer):

    """
    ``Minimizer`` employing Gaussian Process Regression 

    Parameters
    ---------- 
    hypercube : optional, bool
        Boolian toggle for is the n_points should be placed in a latin hypercube, or as a grid across 
        each parameter. Defaults to False
    alpha : float
        Hyperparameter for the fitting, also can represent additional Gaussian noise in measrement points 
    length_scale : float
        Lengthscale parameter for the kernel
    n_points : int
        A number of points which will be measured at, either randomly on a latin hypercube (if hypercube=True) 
        or p^n_points (p = number of parameters) in a grid (if hypercube=False)


    Attributes
    ----------
    history_columns: list[str]
        list of the column titles for the minimizer history
    """


    def __init__(self, parameters, distribution, max_parameter_change, **settings):
        super().__init__(parameters, distribution, max_parameter_change)
        self.hypercube = settings.get('hypercube', False)
        self.alpha = settings.get('alpha', 0.0001)
        self.length_scale = settings.get('length_scale', 0.1)
        n_points = settings.get('n_points', 4)

        self.parameter_names, self.parameter_point_array = self.create_parameter_point_array(parameters, n_points)
        self.change_parameters(parameters)


    def create_parameter_point_array(self, parameters, points=2):
        """
        Takes the constraints of the parameters to be minimised, makes an array of length points
        then performs the Cartesian product, to give every set of coordinates. If no constraints
        are present, then some arbitrary ones are set based on the parameter value.
        
        Parameters
        ----------
        parameters : list
            All ``Parameter`` objects that are being refined.
        points : int
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
        if not self.hypercube:

            for parameter in parameters:
                lower_bound, upper_bound = self.create_bounds(parameter)
                parameter_names.append(str(parameter.name))
                
                bounds_grid = np.linspace(lower_bound, upper_bound, points)
                bounds_array.append(bounds_grid)
            point_array =  list(map(list, itertools.product(*bounds_array))) # * is necessary for unpacking the arrays 
            return parameter_names, point_array
            
        else:
            point_array = []
            samples = st.qmc.LatinHypercube(d=len(parameters), centered=True)
            latin_points = samples.random(n=points)
            for i, parameter in enumerate(parameters):
                lower_bound, upper_bound = self.create_bounds(parameter)
                parameter_names.append(str(parameter.name))
                latin_points[:, i] = self.scale_hypercube(latin_points[:, i], lower_bound, upper_bound)
            return parameter_names, latin_points



    def create_bounds(self, parameter, fraction = 0.2):
        """
        Returns either the parameter constraints (bounds) or some sensible bounds for 
        a given parameter, defaults to +-20% but with 0.1 added/subtracted to account 
        for zero being a possible parameter value

        Parameters
        ----------
        parameter : Parameter instance
            A instance of the MDMC Parameter class
        fraction : optional, float
            The size of the bound, defaults to +-20%
        
        Returns
        -------
        lower_bound : float
            The lower bound for the parameter
        upper_bound : float
            The upper bound for the parameter        
        """
        try:
            lower_bound = parameter.constraints[0]
            upper_bound = parameter.constraints[1]
        except(TypeError):
            if not parameter.value ==0:
                lower_bound = parameter.value*(1.0 - fraction)
                upper_bound = parameter.value*(1.0 + fraction)
            else:
                raise ValueError(f'You have set parameter {parameter.name} value to zero and have no constraints set for it. Please set constraints for it')
        return lower_bound, upper_bound
    
    def scale_hypercube(self, input_array, lower_bound, upper_bound):
        """
        Takes an input array in interval [0,1] and scales the values to instead be in 
        the interval [lower_bound, upper_bound]
        
        Parameters
        ----------
        input_array : array
            The relative tolerance of the convergence check. Defaults to `1e-5`
        lower_bound : float
            The value to scale the array to from 0 
        upper_bound : float
            The value to scale the array to from 1

        Returns
        -------
        array
            Scale array of same shape as input array

        """
        scaled_array = input_array * lower_bound + (upper_bound - lower_bound)
        return scaled_array

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
        change_state = self.comm.bcast(change_state, root=0)
        return True

    @property
    def history_columns(self):

        return ['FoM', 'Change state'] + [p.name for p in self.parameters]

    # pylint: disable=arguments-differ
    # we allow implementations of the abstract method to have different arguments

    def change_parameters(self, parameters):
        """
        Selects a new value for each parameter from the array of parameter values to interrogate

        Parameters
        ----------
        parameters : list
            All ``Parameter`` objects that are being refined
        """
        #self.parameter_names
        for i, parameter_name in enumerate(self.parameter_names):
            point_to_calculate = len(self._history)
            print("history length:"+str(len(self._history)))
            if point_to_calculate <= len(self.parameter_point_array):
                for parameter in self.parameters:
                    if parameter.name == parameter_name:
                        try:
                            parameter.value = self.parameter_point_array[point_to_calculate][i]
                        except(IndexError):
                            parameter.value = self.parameter_point_array[i]
                        print(parameter.name, parameter.value)
                        break
        

    def step(self, FoM):
        """
        Increments the minimization by a step
        """

        self.FoM = FoM
        values = np.array([p.value for p in self.parameters])
        history = [self.FoM]
        history.append('Accepted')
        self.FoM_old = self.FoM
        self.state_changed = True

        history.extend(values)
        self._history.append(history)
        if len(self._history) < len(self.parameter_point_array):
            self.change_parameters(self.parameters)

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
        records = records.astype(dtype=float, errors='ignore')  # Convert to float where possible (i.e. not a string)
        headers = records.columns
        data = records.values

        coordinates = []
        FOMs = []
        for i in range(len(data)):
            coordinate = data[i]
            coordinates.append(coordinate[3:])  # Only append parameters
            FOMs.append(records['FoM'][i])
        
        kernel = kernels.RBF(length_scale = np.ones(len(coordinates[0]))*self.length_scale)
        gpr = GPR(kernel, n_restarts_optimizer=50, alpha = alpha)

        fitted_GPR = gpr.fit(coordinates, FOMs)

        return fitted_GPR, headers

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
            min, max = np.min(column), np.max(column)
            dense_array = np.linspace(min, max, points)
            predictive_coordinates.append(dense_array)

        point_array =  list(itertools.product(*predictive_coordinates))  # predict method needs explicit array

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
        """

        min_coordinates = minimum_position(predicted_FOMs)
        minimum_parameters = measured_parameter_coordinates[min_coordinates]

        return minimum_parameters

    def present_result(self):
        """
        Returns the predicted minimum parameter values and associated figure of merit
        """
        fit, parameter_names = self.GPR_fit()
        points, FoMs = self.GPR_predict(fit)
        minima_coordinate = self.global_minimum_position(FoMs, points)

        return parameter_names, minima_coordinate
