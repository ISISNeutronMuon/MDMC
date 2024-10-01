"""The class for Chi Squared figure of merit calculation with errors"""
import numpy as np

from MDMC.refinement.FoM.FoM_abs import FigureOfMerit, ObservablePair


class ChiSquaredExpError(FigureOfMerit):
    """
    Calculates the figure of merit as a sum of the square difference between
    data points for a single pair of observables, normalised by the errors
    and the number of data points, i.e. the reduced chi-squared.

    Please see the documentation page explanation/figure-of-merit for
    mathematical details.
    """

    def calculate_single_FoM(self, obs_pair: ObservablePair):
        """
        Calculates the chi-squared figure of merit for a single
        pair of observables, potentially rescaled if the experimental
        observable is not on an absolute scale.

        Parameters
        ----------
        obs_pair : ObservablePair
            An ``ObservablePair`` for which the FoM is calculated

        Returns
        -------
        float
            The FoM for the obs_pair
        """


        exp_values = np.array(*obs_pair.exp_obs.dependent_variables.values())
        MD_values = np.array(*obs_pair.MD_obs.dependent_variables.values())
        if obs_pair.auto_scale:

            # Normalise area
            obs_pair.rescale_factor = MD_values.sum() / exp_values.sum()
            # Normalise and remove "zeros"
            # exp_values /= exp_values.max()
            # exp_values[exp_values < 1e-8] = 0.

            # obs_pair.rescale_factor = (np.sum((MD_values / exp_errors) ** 2) / np.sum(
            #     MD_values * exp_values / exp_errors ** 2)) / exp_values.max()

        exp_errors = np.array(*obs_pair.exp_obs.errors.values())
        exp_errors[exp_values < 1e-8] = np.inf
        print(f"{obs_pair.rescale_factor=}")

        norm_factor = self.data_norm_factor(obs_pair=obs_pair)
        value_unreduced = np.sum((obs_pair.calculate_difference() /
                                  exp_errors * obs_pair.rescale_factor) ** 2) / exp_values.size
        return obs_pair.weight * value_unreduced / norm_factor
