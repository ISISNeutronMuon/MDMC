"""The Metropolis-Hastings minimizer class"""
from typing import TYPE_CHECKING, Union, Optional
from pathlib import Path

import numpy as np

from MDMC.refinement.minimizers.minimizer_abs import Minimizer

if TYPE_CHECKING:
    from MDMC.MD import Parameters
    from MDMC.control import Control

class MMC(Minimizer):

    """
    ``Minimizer`` employing the Metropolis-Hastings algorithm

    Parameters
    ----------
    control: Control
        The ``Control`` object which uses this Minimizer.
    MC_norm : float
        Normalization parameter for MC which determines the accept/reject ratio, default is 1.0
    max_parameter_change: float, optional
        Maximum factor by which a Parameter can change each step of the
        refinement. Defaults to `0.01`
    conv_tol : float, optional
        The relative tolerance of the convergence check. Defaults to `1e-5`
    min_steps : int, optional
        The number of refinement steps with an accepted state change after which
        convergence is checked. If the number of accepted state changes is less than this,
        then the refinement is deemed as not converged. Defaults to `2`
    distribution : str, optional
        The distribution from which ``Parameter`` changes are selected. Defaults to 'uniform'

    Attributes
    ----------
    history_columns: list[str]
        list of the column titles for the minimizer history
    """

    DISTRIBUTION = {'uniform': np.random.uniform}

    def __init__(self, control: 'Control', parameters: 'Parameters', \
        previous_history: Optional[Union[Path, str]] = None,**settings: dict):
        super().__init__(control, parameters, previous_history)
        self.MC_norm = settings.get('MC_norm', 1.0)

        self.parameters = parameters
        self.max_parameter_change = settings.get('max_parameter_change', 0.01)
        self.conv_tol = settings.get('conv_tol', 1e-5)
        self.min_steps = settings.get('min_steps', 2)
        distribution = 'uniform'
        self.distribution = self.__class__.DISTRIBUTION[distribution]

        self.previous_history = previous_history
        self.state_changed = False

    @property
    def history_columns(self) -> 'list[str]':
        """
        Returns column labels of the history

        Returns
        -------
        list[str]
            A ``list`` of ``str`` containing all the column labels in the history
        """
        return ['FoM', 'Change state'] + list(self.parameters)

    def step(self, FoM: float) -> None:
        """
        Increments the minimization by a step

        Parameters
        ----------
        FoM : float
            The current figure of merit value.
        """

        self.FoM = FoM
        parameters = {p: self.parameters[p].value for p in self.parameters}
        history = [self.FoM]

        if self.change_state():
            history.append('Accepted')
            self.FoM_old = self.FoM
            self.parameters_old_values = parameters
            self.state_changed = True

        else:
            history.append('Rejected')
            self.FoM = self.FoM_old
            self.reset_parameters()
            self.state_changed = False

        history.extend(list(parameters.values()))
        self._history.append(history)
        self.change_parameters()

    def change_state(self) -> bool:
        """
        Stochastic determination of whether the state should change based on the
        FoM

        Returns
        -------
        bool
            `True` if the state should be change
        """

        # Only determine if state will be changed
        prob = min(1, np.exp((self.FoM_old - self.FoM) / self.MC_norm))
        change_state = bool(prob > np.random.random())

        return change_state

    def change_parameters(self) -> None:
        """
        Selects a new value for each parameter from a distribution centered
        around the current value.

        Note that for ``Parameter``s with ``constraints`` set, any proposed new value that would
        lie outside the range of the constraint is clipped to the lower or upper limit as
        appropriate.

        Parameters
        ----------
        parameters : Parameters
            All ``Parameter`` objects that are being refined
        """

        # Calculate magnitude of parameter changes
        # Faster to generate all random numbers at once
        changes = self.distribution(-self.max_parameter_change,
                                    self.max_parameter_change,
                                    len(self.parameters))

        for i, parameter in enumerate(self.parameters.values()):
            new_value = parameter.value * (1 + changes[i])
            # If the parameter is constrained, then clip changes that would be out of range
            if parameter.constraints is not None:
                if new_value < parameter.constraints[0]:
                    new_value = parameter.constraints[0]
                elif new_value > parameter.constraints[1]:
                    new_value = parameter.constraints[1]

            self.parameters[parameter.name].value = new_value

    def has_converged(self) -> bool:
        """
        Checks if the refinement process has converged on a stable solution.
        Specifically, it checks if the Figure of Merit and the parameters being refined have all
        changed less than the relative conversion tolerance (`conv_tol`) between the last two
        accepted refinement steps. It also allows specifying a minimum number of refinement
        steps (`min_steps`) that must have been accepted before checking for convergence.

        Returns
        -------
        bool
            Whether or not the minimizer has converged.
        """

        # select the history of accepted state changes
        accepted_history = (self.history['Change state'] == 'Accepted')
        accepted_history = self.history[accepted_history]
        if len(accepted_history) >= self.min_steps:
            # drop 'Change state' column to select only parameters;
            # turn to np.array for easy slicing
            param_history = np.array(
                accepted_history.drop('Change state', axis=1))
            converged = np.allclose(
                param_history[-1], param_history[-2], rtol=self.conv_tol)
        else:
            converged = False

        return converged

    def reset_parameters(self) -> None:
        """
        Resets the ``Parameter`` values to the values from the previous MMC step
        """

        for parameter in self.parameters:
            self.parameters[parameter].value = self.parameters_old_values[parameter]

    def extract_result(self) -> 'list[str]':
        """
        Extracts the result data from the history of the minimizer run

        Returns
        -------
        output_data: list[str]
            A list of: last accepted point coordinates, last accepted point FoM value,
            best point coordinates, best point FoM
        """
        self.reset_parameters()
        history = self.history

        last_param_row = history.iloc[-1]
        last_FoM_value = last_param_row[0]

        # Find lowest parameters & FoM
        lowest_FoM_id = history["FoM"].idxmin()
        lowest_FoM_row = history.iloc[lowest_FoM_id]
        lowest_FoM_value = lowest_FoM_row.get("FoM")

        last_param_row = last_param_row.drop("FoM").drop("Change state")
        lowest_FoM_row = lowest_FoM_row.drop("FoM").drop("Change state")

        last_parameters_found = tuple(last_param_row)
        lowest_FoM_parameters = tuple(lowest_FoM_row)

        output_data = [last_parameters_found, last_FoM_value,
                       lowest_FoM_parameters, lowest_FoM_value]

        return output_data

    def format_result_string(self, minimizer_output: list) -> str:
        """
        Formats a string output for the results of an MMC minimizer run

        Parameters
        ----------
        minimizer_output: list
            A list of: last accepted point coordinates, last accepted point FoM value,
            best point coordinates, best point FoM

        Returns
        -------
        output_string: str
            A string containing the following: whether the minimizer has converged, last parameters,
            last FoM value, optimal (lowest FoM) parameters, optimal (lowest) FoM value
        """
        if self.has_converged():
            converged_message = '\nThe refinement has converged.'
        else:
            converged_message = "\nThe refinement has not converged."

        output_string = (f'{converged_message} \n \n'
                         f'Last accepted point is: \n'
                         f'{minimizer_output[0]} with a minimum '
                         f'FoM of {minimizer_output[1]}. \n \n'
                         f'Best point measured was: \n'
                         f'{minimizer_output[2]} for a minimum FoM of '
                         f'{minimizer_output[3]}.\n \n ')

        return output_string
