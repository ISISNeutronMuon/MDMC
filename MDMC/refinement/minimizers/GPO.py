"""The Gaussian-Process-Optimizer minimizer class"""
import itertools
from typing import Optional, Tuple

import numpy as np
import scipy.stats as st
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor as skGPR
from sklearn.gaussian_process import kernels
from skopt import Optimizer
from scipy.ndimage import minimum_position, minimum

from MDMC.refinement.minimizers.minimizer_abs import Minimizer
from MDMC.refinement.minimizers.GPR import GPR
from MDMC.MD.parameters import Parameters, Parameter


class GPO(Minimizer):
    """
    ``Minimizer`` which uses Gaussian process regression to find the global minimum
    figure of merit. The optimizer comes from scikit-optimize
    https://scikit-optimize.github.io/stable/modules/generated/skopt.optimizer.Optimizer.html
    It acts in an ask/tell arcitecture, where the optimizer is "asked" for the best
    parameter values to measure at, then when the measurement is complete, we "tell"
    the optimizer what the result was and it updates its model. The optimizer
    is configured to cycle between prioritising exploration of the space, and
    exploitation of the minima, in order to find the global minimum, without becoming
    stuck in a local minimum.

    Parameters
    ----------
    n_points: int
        The number of points to measure (in an ask/tell manner)

    Attributes
    ----------
    history_columns: list[str]
        list of the column titles, and parameter names in the minimizer history
    """


    def __init__(self, parameters, n_points, **settings):
        super().__init__(parameters, n_points)

        self.parameters = parameters
        self.n_points = n_points
        # Ensure all parameters have bounds
        self.parameter_bounds = np.array(tuple(GPR.create_bounds(p) for p in self.parameters))


        # Initialise the optimizer, use Gaussian process estimator, an acquisition function which
        # switches between exploration and exploitation, a sampling acquisition optimizer, and
        # a latin hypercube for determining the positions of the inital 20 points (before points
        # are decided based on the best position as determined by the Gaussian process).
        self.optimizer =Optimizer(self.parameter_bounds,"GP", acq_func="gp_hedge",
                acq_optimizer="sampling", initial_point_generator="lhs", n_initial_points=20)


    @property
    def history_columns(self) -> list[str]:

        return ['FoM', 'Change state', 'Predicted min coords', 'Predicted min FoM'] + list(self.parameters)

    def has_converged(self) -> bool:
        """
        Checks if the refinement process has finished, i.e. if the number of points
        equal to n_points have been measured.

        Returns
        -------
        bool
            Whether or not the minimizer has converged.
        """
        return len(self.history) >= self.n_points

    def GP_opt_tell(self, measured_values: np.ndarray, measured_FoM: float) -> None:
        """Updates the Gaussian process optimizer with next measured point"""
        self.optimizer.tell(measured_values, measured_FoM)


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
        if point_to_calculate <= self.n_points:
            coordinates = self.optimizer.ask()
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
        self.predicted_FoM = self.optimizer.get_result()['fun']
        self.predicted_min_pos = self.optimizer.get_result()['x']
        history = [self.FoM, 'Accepted', self.predicted_min_pos, self.predicted_FoM]
        self.state_changed = True

        values = np.array([self.parameters[p].value for p in self.parameters])
        history.extend(values)
        self._history.append(history)
        if not self.has_converged():
            self.change_parameters()

    def present_result(self) -> str:
        """
        Sets the parameters to those predicted to return the minimum FoM, returns
        the coordinates of the minima and the predicted FoM.

        Returns
        -------
        output_string : str
            A string presenting the parameters for which the calculated and predicted
            figures of merit are lowest. To be printed by Control to the user.
        """
        FoMs = [FoM[:][0] for FoM in self._history]
        min_FOM_measured = np.min(FoMs)
        min_parameters_measured = self._history[np.where(FoMs==min_FOM_measured)[0]]

        output_string = (f'Best point measured was \n'
            f'{min_parameters_measured} for a minimum FoM of '
            f'{min_FOM_measured}. \n\n '
            f'Predicted minimum coordinate is {self.predicted_min_pos} for a minimum '
            f'FoM of {self.predicted_FoM}. \n ')

        return output_string