"""The Gaussian-Process-Optimizer minimizer class"""
from typing import TYPE_CHECKING
import numpy as np

from skopt import Optimizer

from MDMC.refinement.minimizers.minimizer_abs import Minimizer
from MDMC.refinement.minimizers.GPR import GPR

if TYPE_CHECKING:
    from MDMC.MD.parameters import Parameters

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
    stuck in a local minimum. The first 20 points will be spaced according to a latin
    hypercube, to cover the available space, subsequent points will then be chosen according
    to the acquisition function and the measured values.
    Due to the potential large jumps between the points, a reasonable amount of equlibration
    of the MD simulation is likely required.
    This optimizer is likely to be the fastest converging (fewest MD steps) option for MDMC.

    Parameters
    ----------
    parameters: Parameters
        The parameters in the simulation Universe to be optimized

    Settings
    ----------
    n_points: int
        The number of points to measure

    Attributes
    ----------
    history_columns: list[str]
        list of the column titles, and parameter names in the minimizer history
    """


    def __init__(self, parameters: 'Parameters', **settings: dict):
        super().__init__(parameters)

        self.parameters = parameters
        self.n_points = settings.get('n_points', 20)
        self.predicted_FoM = 1e9
        self.predicted_min_pos = []
        # Ensure all parameters have bounds
        self.parameter_bounds = [tuple(GPR.create_bounds(parameter)) \
                                for parameter in parameters.values()]

        self.parameter_names =  [str(name) for name in parameters.keys()]

        np.random.seed(7) # This should mean results are reproducible in tests

        # Initialise the optimizer, use Gaussian process estimator, an acquisition function which
        # switches between exploration and exploitation, a sampling acquisition optimizer, and
        # a latin hypercube for determining the positions of the inital 20 points (before points
        # are decided based on the best position as determined by the Gaussian process).
        self.optimizer = Optimizer(self.parameter_bounds,"GP", acq_func="gp_hedge",
                acq_optimizer="sampling", initial_point_generator="lhs", n_initial_points=20)


    @property
    def history_columns(self) -> list[str]:

        return ['FoM', 'Change state', 'Pred coords', 'Pred FoM'] + list(self.parameters)


    def has_converged(self) -> bool:
        """
        Checks if the refinement process has finished, i.e. if the number of points
        equal to or greater than n_points have been measured.

        Returns
        -------
        bool
            Whether or not the minimizer has converged.
        """
        return len(self.history) >= self.n_points


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

        if len(self._history) <= self.n_points:
            coordinates = self.optimizer.ask()
            self.set_parameter_values(self.parameter_names, coordinates)


    def reset_parameters(self) -> None:
        """Not necessary for this minimizer"""
        # pylint: disable=unnecessary-pass
        pass

    def step(self, FoM: float) -> None:
        """
        Increments the minimization by a step, tells the optimizer the most recent measured point
        asks for the coordinates of the next point, updates the history, checks for convergance
        and then changes parameters if an additional step is required.

        Parameters
        ----------
        FoM : float
            The current figure of merit value.
        """

        self.FoM = FoM
        values = list((self.parameters[p].value for p in self.parameters))

        self.optimizer.tell(values, float(self.FoM))

        self.predicted_FoM = self.optimizer.get_result()['fun']
        self.predicted_min_pos = self.optimizer.get_result()['x']
        history = [self.FoM, 'Accepted', self.predicted_min_pos, self.predicted_FoM]
        self.state_changed = True

        history.extend(values)
        self._history.append(history)
        if not self.has_converged():
            self.change_parameters()


    def present_result(self) -> str:
        """
        Returns the coordinates of the minima and the predicted FoM.

        Returns
        -------
        output_string : str
            A string presenting the parameters for which the calculated and predicted
            figures of merit are lowest. To be printed by Control to the user.
        """
        FoMs = [FoM[:][0] for FoM in self._history]
        min_FOM_measured = np.min(FoMs)
        min_parameters_measured = self._history[np.where(FoMs==min_FOM_measured)[0][0]][3]
        # the [0][0][3] is just to get the parameters from the _history

        output_string = (f'Best point measured was \n'
            f'{min_parameters_measured} for a minimum FoM of '
            f'{min_FOM_measured}. \n\n '
            f'Predicted minimum coordinate is {self.predicted_min_pos} for a minimum '
            f'FoM of {self.predicted_FoM}. \n ')

        return output_string
