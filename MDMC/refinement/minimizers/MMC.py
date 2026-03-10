"""The Metropolis-Hastings minimizer class"""

from collections.abc import Sequence
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Optional, Union

import cma
import numpy as np

from MDMC.refinement.minimizers.minimizer_abs import Minimizer

if TYPE_CHECKING:
    from MDMC.control import Control
    from MDMC.MD import Parameters


class MMC(Minimizer):
    """
    Minimiser using CMA-ES, but using it sequentially.

    Normally, CMA-ES produces several sets of input parameters per batch/generation.
    The MMC wrapper executes them one at a time, and asks the CMA-ES optimiser
    for a new batch of inputs every time the existing batch has been used up.

    Parameters
    ----------
    control: Control
        The ``Control`` object which uses this Minimizer.
    max_parameter_change: float, optional
        Maximum factor by which a Parameter can change each step of the
        refinement. Defaults to `0.01`
    conv_tol : float, optional
        The relative tolerance of the convergence check. Defaults to `1e-5`
    min_steps : int, optional
        The number of refinement steps with an accepted state change after which
        convergence is checked. If the number of accepted state changes is less than this,
        then the refinement is deemed as not converged. Defaults to `2`

    Attributes
    ----------
    history_columns: list[str]
        list of the column titles for the minimizer history
    """

    DISTRIBUTION = {"uniform": np.random.uniform}

    def __init__(
        self,
        control: "Control",
        parameters: "Parameters",
        previous_history: Optional[Union[Path, str]] = None,
        **settings: dict,
    ):
        super().__init__(control, parameters, previous_history)

        self.parameters = parameters
        self.max_parameter_change = settings.get("max_parameter_change", 0.2)
        self.conv_tol = settings.get("conv_tol", 1e-4)
        self.min_steps = settings.get("min_steps", 2)

        self.previous_history = previous_history
        self.state_changed = False
        self.optimiser = cma.CMAEvolutionStrategy(
            [par.value for par in self.parameters.values()],
            self.max_parameter_change,
            {
                "bounds": [
                    [par.constraints[0] for par in self.parameters.values()],
                    [par.constraints[1] for par in self.parameters.values()],
                ],
                "CMA_elitist": True,
                "tolfun": self.conv_tol * 100,
                "tolx": 1e-3,
                "tolfunhist": self.conv_tol * 10,
            },
        )
        self.new_parameters = self.optimiser.ask()
        self.used_parameters, self.used_values = [], []

    @property
    def history_columns(self) -> "list[str]":
        """
        Returns column labels of the history

        Returns
        -------
        list[str]
            A ``list`` of ``str`` containing all the column labels in the history
        """
        return ["FoM", "Change state"] + list(self.parameters)

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
        self.used_parameters.append([val for val in parameters.values()])
        self.used_values.append(FoM)

        history.append("Accepted")
        self.FoM_old = self.FoM
        self.parameters_old_values = parameters
        self.state_changed = True

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
        return True

    def next_parameter_point(self) -> Sequence[float]:
        """Return the next set of simulation parameters.

        If the current batch has been exhausted, it generates a new batch using CMA-ES."""
        if not self.new_parameters:
            self.optimiser.tell(self.used_parameters, self.used_values)
            self.new_parameters = self.optimiser.ask()
            self.used_parameters = []
            self.used_values = []

        return self.new_parameters.pop()

    def change_parameters(self) -> None:
        """Assign new values to the simulation parameters."""

        new_values = self.next_parameter_point()

        for i, parameter in enumerate(self.parameters.values()):
            self.parameters[parameter.name].value = new_values[i]

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
        if len(self.history) <= self.min_steps:
            return False
        param_history = np.array(self.history.drop("Change state", axis=1))
        converged = np.allclose(param_history[-1], param_history[-2], rtol=self.conv_tol)
        return self.optimiser.stop() or converged

    def reset_parameters(self) -> None:
        """
        Not used.
        """
        pass

    def extract_result(self) -> "list[str]":
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

        output_data = [
            last_parameters_found,
            last_FoM_value,
            lowest_FoM_parameters,
            lowest_FoM_value,
        ]

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
            converged_message = "The refinement has converged."
        else:
            converged_message = "The refinement has not converged."

        # as of numpy 2.0.0, np.float64 has repr e.g. "np.float64(3.0)" instead of "3.14"
        # we use legacy print options to make the string nicer with less fiddling
        with np.printoptions(legacy="1.25"):
            output_string = f"""
                            {converged_message}

                            Last accepted point is:
                            {minimizer_output[0]} with a minimum
                            FoM of {minimizer_output[1]}.

                            Best point measured was:
                            {minimizer_output[2]} for a minimum FoM of
                            {minimizer_output[3]}.
                            """

            return dedent(output_string)
