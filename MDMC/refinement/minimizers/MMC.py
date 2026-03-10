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

    DISTRIBUTION = {"uniform": np.random.uniform}

    def __init__(
        self,
        control: "Control",
        parameters: "Parameters",
        previous_history: Optional[Union[Path, str]] = None,
        **settings: dict,
    ):
        super().__init__(control, parameters, previous_history)
        self.MC_norm = settings.get("MC_norm", 1.0)

        self.parameters = parameters
        self.max_parameter_change = settings.get("max_parameter_change", 0.01)
        self.conv_tol = settings.get("conv_tol", 1e-5)
        self.min_steps = settings.get("min_steps", 2)
        distribution = "uniform"
        self.distribution = self.__class__.DISTRIBUTION[distribution]

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
                "tolfun": self.conv_tol*10,
                "tolx": 1e-3,
                "tolfunhist": self.conv_tol,
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
        if not self.new_parameters:
            self.optimiser.tell(self.used_parameters, self.used_values)
            self.new_parameters = self.optimiser.ask()
            self.used_parameters = []
            self.used_values = []

        return self.new_parameters.pop()

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
        return self.optimiser.stop()

    def reset_parameters(self) -> None:
        """
        Resets the ``Parameter`` values to the values from the previous MMC step
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
