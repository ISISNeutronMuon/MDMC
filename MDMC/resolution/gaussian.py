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

"""The Gaussian resolution subclass"""

import numpy as np

from MDMC.common.constants import h_bar
from MDMC.common.resolution_functions import gaussian
from MDMC.resolution.resolution import Resolution


class GaussianResolution(Resolution):
    """
    A class for applying a Gaussian resolution to an :any:`Observable`.
    """

    def __init__(self, e_res: float):
        # converts energy resolution from user-friendly ueV to system unit meV
        self.e_res = e_res / 1000

    def apply(self, FQt, t, Q):
        N_Q, N_T = np.shape(FQt)
        window = self.window_in_t(t[:N_T])

        return np.broadcast_to(window, (N_Q, N_T)) * FQt

    def window_in_w(self, w: np.ndarray, mu: float = 0.0, norm: bool = True) -> np.ndarray:
        """
        Calculate the Gaussian window function in frequency space.

        Parameters
        ----------
        w : ~numpy.ndarray
            An array of frequency points.
        mu : float
            The offset of the function (defaults to 0).
        norm : bool
            If ``True``, normalises the distribution to unity.

        Returns
        -------
        ~numpy.ndarray
            The window function in frequency space (i.e. the Gaussian with
            FWHM self.e_res, centred on 0) over the frequency array `w`.
        """

        window = gaussian(w, self.e_res, mu, norm)

        return window

    def window_in_t(self, t: np.ndarray) -> np.ndarray:
        """
        Calculate the Gaussian window function in time space.

        Parameters
        ----------
        t : ~numpy.ndarray
            An array of time points.

        Returns
        -------
        ~numpy.ndarray
            The Gaussian window over the times in the array `t`.
        """

        # We convert the FWHM energy resolution (in meV) into sigma_t (in fs) using the inverse
        # relationship between the width of a Gaussian and its Fourier transform,
        # rather than explicitly transforming it, then applying a factor
        # of 1e18 to convert from h / h_bar's units of eV s into system units.

        sigma_t = (2 * np.sqrt(2 * np.log(2)) * h_bar * 1e18) / self.e_res
        window = gaussian(t, sigma_t, norm=False)

        return window

    def __repr__(self):
        """
        Represent a ``GaussianResolution`` object as the given FWHM energy resolution.
        """

        return "Resolution" + str({"gaussian": self.e_res})
