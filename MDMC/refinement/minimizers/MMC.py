"""The Metropolis-Hastings minimizer class"""
import numpy as np
from MDMC.refinement.minimizers.minimizer_abs import Minimizer

class MMC(Minimizer):

    """
    ``Minimizer`` employing the Metropolis-Hastings algorithm
    """

    @property
    def history_columns(self):

        return ['FoM', 'Change state'] + [p.name for p in self.parameters]

    def step(self, FoM):

        """
        Increments the minimization by a step
        """

        self.FoM = FoM
        values = np.array([p.value for p in self.parameters])
        history = [self.FoM]

        if self.change_state():
            history.append('Accepted')
            self.FoM_old = self.FoM
            self.parameters_old_values = values
            self.state_changed = True

        else:
            history.append('Rejected')
            self.FoM = self.FoM_old
            self.reset_parameters()
            self.state_changed = False

        history.extend(values)
        self._history.append(history)
        self.change_parameters(self.parameters)

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
            change_state = True if prob > np.random.random() else False
        else:
            change_state = None
        # Broadcast to all processes whether or not the state will be changed
        change_state = self.comm.bcast(change_state, root=0)

        return change_state

    def change_parameters(self, parameters):

        """
        Selects a new value for each parameter from a distribution centered
        around the current value.

        Note that for ``Parameter``s with ``constraints`` set, any proposed new value that would
        lie outside the range of the constraint is clipped to the lower or upper limit as
        appropriate.

        Parameters
        ----------
        parameters : list
            All ``Parameter`` objects that are being refined
        """

        # Only calculate magnitude of parameter changes on rank 0 process, so
        # that each process ends up with same parameters
        if self.comm.rank == 0:
            # Faster to generate all random numbers at once
            changes = self.distribution(-self.max_parameter_change,
                                        self.max_parameter_change,
                                        len(parameters))
        else:
            changes = None
        # Broadcast parameters changes to all processes
        changes = self.comm.bcast(changes, root=0)
        # Change parameters by same amount on all processes
        for i, parameter in enumerate(parameters):
            new_value = parameter.value * (1 + changes[i])
            # If the parameter is constrained, then clip changes that would be out of range
            if parameter.constraints is not None:
                if new_value < parameter.constraints[0]:
                    new_value = parameter.constraints[0]
                elif new_value > parameter.constraints[1]:
                    new_value = parameter.constraints[1]

            parameter.value = new_value

    def reset_parameters(self):

        """
        Resets the ``Parameter`` values to the values from the previous MMC step
        """

        for i, parameter in enumerate(self.parameters):
            parameter.value = self.parameters_old_values[i]
