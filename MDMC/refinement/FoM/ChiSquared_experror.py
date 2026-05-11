# MDMC is a package for the optimisation of classical potentials with experimental data
# Copyright (C) 2026 MDMC Developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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

        norm_factor = self.data_norm_factor(obs_pair=obs_pair)
        obs_pair.fom_contribution = (
            obs_pair.interpolate_MD_onto_exp() / obs_pair.calculate_exp_errors()
        ) ** 2
        value_unreduced = np.nansum(
            (obs_pair.interpolate_MD_onto_exp() / obs_pair.calculate_exp_errors()) ** 2
        )
        print(f"ChiSquared_noneerror.calculate_single_FoM: norm_factor: {norm_factor}")
        print(f"ChiSquared_noneerror.calculate_single_FoM: value_unreduced: {value_unreduced}")

        if obs_pair.auto_scale:
            exp_errors = np.array(*obs_pair.exp_obs.errors.values())
            exp_values = np.array(*obs_pair.exp_obs.dependent_variables.values())
            MD_values = np.array(*obs_pair.matching_obs.dependent_variables.values())
            A = np.sum((MD_values / exp_errors) ** 2)
            B = np.sum(MD_values * exp_values / exp_errors**2)
            print(f"ChiSquared_noneerror.calculate_single_FoM: A: {A}")
            print(f"ChiSquared_noneerror.calculate_single_FoM: B: {B}")
            obs_pair.rescale_factor = A / B
            print(f"Rescale factor: {obs_pair.rescale_factor}")

        return obs_pair.weight * value_unreduced / norm_factor
