"""The Gaussian-Process-Optimizer minimizer class"""
from typing import TYPE_CHECKING
import numpy as np
import pickle
from skopt import Optimizer

from MDMC.refinement.minimizers.minimizer_abs import Minimizer
from MDMC.refinement.minimizers.GPR import GPR

if TYPE_CHECKING:
    from MDMC.MD.parameters import Parameters
    from MDMC.control import Control


class GPO(Minimizer):
    """
    ``Minimizer`` which uses Gaussian process regression to find the global minimum
    figure of merit. The optimizer comes from scikit-optimize
    https://scikit-optimize.github.io/stable/modules/generated/skopt.optimizer.Optimizer.html
    It acts in an ask/tell architecture, where the optimizer is "asked" for the best
    parameter values to measure at, then when the measurement is complete, we "tell"
    the optimizer what the result was and it updates its model. The optimizer
    is configured to cycle between prioritising exploration of the space, and
    exploitation of the minima, in order to find the global minimum, without becoming
    stuck in a local minimum. The first ``n_initial`` points will be spaced according to a latin
    hypercube, to cover the available space, subsequent points will then be chosen according
    to the acquisition function and the measured values.
    Due to the potential large jumps between the points, a reasonable amount of equlibration
    of the MD simulation is likely required.
    This optimizer is likely to be the fastest converging (fewest MD steps) option for MDMC.

    Parameters
    ----------
    control: Control
        The ``Control`` object which uses this Minimizer.
    parameters: Parameters
        The parameters in the simulation Universe to be optimized

    Settings
    ----------
    n_initial: int, optional
        The number of points used for the initial latin hypercube coverage of the parameter
        space. Optional. If no value is given it defaults to 20. Note that if the
        associated ``Control`` objects has a maximum number of refinement steps (defined in
        ``Control.n_steps``) which is smaller than ``n_initial`` then that value will be
        used instead.

    Attributes
    ----------
    history_columns: list[str]
        list of the column titles, and parameter names in the minimizer history
    """

    def __init__(self, control: 'Control', parameters: 'Parameters', **settings: dict):
        super().__init__(control, parameters)
       
        self._progress_file = "gpo_step.pkl"
        if os.path.exists(self._progress_file):
            self.load_progress(self._progress_file)
        else:
            self.optimizer()
            
        self.parameters = parameters
        self.n_initial = settings.get('n_initial', 20)
        if self.control.n_steps:
            self.n_initial = min(self.control.n_steps, self.n_initial)
        self.predicted_FoM = 1e9
        self.predicted_min_pos = []
        # Ensure all parameters have bounds
        self.parameter_bounds = [tuple(GPR.create_bounds(parameter)) \
                                for parameter in parameters.values()]

        self.parameter_names = [str(name) for name in parameters.keys()]

        np.random.seed(7) # This should mean results are reproducible in tests

        # Initialise the optimizer, use Gaussian process estimator, an acquisition function which
        # switches between exploration and exploitation, a sampling acquisition optimizer, and
        # a latin hypercube for determining the positions of the inital 20 points (before points
        # are decided based on the best position as determined by the Gaussian process).
        self.optimizer = Optimizer(self.parameter_bounds,"GP", acq_func="gp_hedge",
                                   acq_optimizer="sampling", initial_point_generator="lhs",
                                   n_initial_points=self.n_initial, model_queue_size=1)


        def refine(self, n_points):
            if n_points > self.n_initial:
                self.n_initial = n_points
                self.optimizer.n_initial_points = self.n_initial

            for _ in range(n_points):
                next_point = self.optimizer.ask()
                measured_points = self.perform_measurement(next_point)
                self.optimizer.tell(next_point, measured_points)
                self.save_progress(self._progress_file)

        def save_progress(self, filename):
            with open(filename, "wb") as f:
                pickle.dump(self.optimizer, f)

        def load_progress(self, filename):
            with open(filename, "rb") as f:
                self.optimizer = pickle.load(f)

    @property
    def history_columns(self) -> 'list[str]':
        """
        Returns column labels of the history

        Returns
        -------
        list[str]
            A ``list`` of ``str`` containing all the column labels in the history
        """

        return ['FoM', 'Change state', 'Pred coords', 'Pred FoM'] + list(self.parameters)

    def has_converged(self) -> bool:
        """
        Checks if the refinement process has finished, i.e. if the number of points is
        equal to or greater than the number of maximum refinement points of the associated
        ``Control`` object.

        Returns
        -------
        bool
            Whether or not the minimizer has converged.
        """
        return len(self.history) >= self.control.n_steps

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

        if not self.has_converged():
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

    def extract_result(self) -> list:
        """
        Extracts the measured & predicted FoM and point(s)

        Returns
        -------
        list_of_outputs: list
            A list of: coordinates of lowest FoM, Minimum FoM, Coordinate of best predicted FoM,
            Minimum predicted FoM
        """
        FoMs = [FoM[:][0] for FoM in self._history]
        min_FoM_measured = np.min(FoMs)
        min_parameters_measured = self._history[np.where(FoMs == min_FoM_measured)[0][0]][4:]
        # the [0][0][4:] is just to get the parameters from the _history

        return [
            tuple(min_parameters_measured),
            float(min_FoM_measured),
            tuple(self.predicted_min_pos),
            self.predicted_FoM
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
        output_string: str
            An output string, formatted with the appropriate information about measured
            and predicted points
        """

        if self.has_converged():
            converged_message = '\nThe refinement has finished.'
        else:
            converged_message = "\nThe refinement has not finished."

        output_string = (f'{converged_message} \n \n'
                         f'Minimum measured point is: \n'
                         f'{minimizer_output[0]} with an '
                         f'FoM of {minimizer_output[1]}. \n \n'
                         f'Minimum point predicted is: \n'
                         f'{minimizer_output[2]} for an '
                         f'FoM of {minimizer_output[3]}.\n \n ')

        return output_string
