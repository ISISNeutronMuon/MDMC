"""The Metropolis-Hastings minimizer class"""
import numpy as np

from MDMC.refinement.minimizers.minimizer_abs import Minimizer


class MMC(Minimizer):

    """
    ``Minimizer`` employing the Metropolis-Hastings algorithm

    Parameters
    ----------
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

    def __init__(self, parameters, **settings):
        super().__init__(parameters)
        self.MC_norm = settings.get('MC_norm', 1.0)

        self.parameters = parameters
        self.max_parameter_change = settings.get('max_parameter_change', 0.01)
        self.conv_tol = settings.get('conv_tol', 1e-5)
        self.min_steps = settings.get('min_steps', 2)
        # Parameters are only changed by rank 0 process, and so only rank 0
        # Minimizer needs a random distribution
        distribution = 'uniform'
        if self.comm.rank == 0:
            self.distribution = self.__class__.DISTRIBUTION[distribution]
        else:
            self.distribution = None

    @property
    def history_columns(self):

        return ['FoM', 'Change state'] + list(self.parameters)


    def step(self, FoM: float) -> None:
        """Increments the minimization by a step"""

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

    def change_state(self):
        """
        Stochastic determination of whether the state should change based on the
        FoM

        Returns
        -------
        bool
            `True` if the state should be change
        """

        # Only determine if state will be changed on rank 0 process
        if self.comm.rank == 0:
            prob = min(1, np.exp((self.FoM_old - self.FoM) / self.MC_norm))
            change_state = bool(prob > np.random.random())
        else:
            change_state = None
        # Broadcast to all processes whether or not the state will be changed
        change_state = self.comm.bcast(change_state, root=0)

        return change_state

    def change_parameters(self):
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

        # Only calculate magnitude of parameter changes on rank 0 process, so
        # that each process ends up with same parameters
        if self.comm.rank == 0:
            # Faster to generate all random numbers at once
            changes = self.distribution(-self.max_parameter_change,
                                        self.max_parameter_change,
                                        len(self.parameters))
        else:
            changes = None
        # Broadcast parameters changes to all processes
        changes = self.comm.bcast(changes, root=0)
        # Change parameters by same amount on all processes
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
        changed less than the relative conversion tolerance (`conv_tol`) between the
        last two accepted refinement steps. It also allows specifying a minimum
        number of refinement steps (`min_steps`) that must have been accepted
        before checking for convergence.

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

    def reset_parameters(self):
        """
        Resets the ``Parameter`` values to the values from the previous MMC step
        """

        for parameter in self.parameters:
            self.parameters[parameter].value = self.parameters_old_values[parameter]

    def present_result(self):
        """
        Sets the parameters to those predicted to return the last FoM, returns
        coordinates of the minima and the predicted FoM.

        Returns
        -------
        output_string : str
            A string presenting the best measured parameters and the current ones, to be printed
            by Control to the user.
        """
        self.reset_parameters()
        final_coordinate = np.array([self.parameters[p].value for p in self.parameters])
        coordinate_names = list(p for p in self.parameters)

        minimum_FoM, min_index = self.history['FoM'].min(), self.history['FoM'].idxmin()
        minimum_parameters = self.history.iloc[min_index,2:]

        output_string = (f'Refined parameters {coordinate_names} have converged '
        f'coordinate: {final_coordinate}\n with a FoM of {self.FoM}.\n '
        f'Best point measured was \n {minimum_parameters} \n '
        f'for a minimum FoM of {minimum_FoM}.')

        return output_string
