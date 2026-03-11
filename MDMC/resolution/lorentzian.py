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

"""The Lorentzian resolution subclass"""
import numpy as np

from MDMC.common.resolution_functions import lorentzian
from MDMC.resolution.resolution import Resolution


class LorentzianResolution(Resolution):
    """
    A class for applying a Lorentzian resolution to an :any:`Observable`.
    """

    def __init__(self, e_res: float):
        self.e_res = e_res / 1000

    def apply(self, FQt, t, Q):
        N_Q, N_T = np.shape(FQt)
        window = self.window_in_t(t[:N_T])

        return np.broadcast_to(window, (N_Q, N_T)) * FQt

    def window_in_w(self, w: np.ndarray) -> np.ndarray:
        """
        Calculate the Lorentzian window function in frequency space.

        Parameters
        ----------
        w : ~numpy.ndarray
            An array of frequencies.

        Returns
        -------
        ~numpy.ndarray
            The Lorentzian window over the frequency array `w`.
        """

        window = lorentzian(w, self.e_res)

        return window

    def window_in_t(self, t: np.ndarray) -> np.ndarray:
        """
        Calculate the Lorentzian window function in time space.

        Parameters
        ----------
        t : ~numpy.ndarray
            An array of time points.

        Returns
        -------
        ~numpy.ndarray
            The Lorentzian window over the time array `t`.
        """
        # The Lorentzian window in time space (i.e. the Fourier transform centred around zero)
        # The Fourier transform of the Lorentzian is
        # F(k) = e^((2 * pi * i) * k * x_0) - Gamma * pi * |k|
        # where x_0 is the offset and Gamma the FWHM.
        # thus as the instrument resolution function is centred around x_0, this simplifies to
        # e^(-Gamma * pi * |k|).

        window = np.exp(-self.e_res * np.pi * np.abs(t))

        return window

    def __repr__(self):
        """
        Represent a ``LorentzianResolution`` object as the given FWHM energy resolution.
        """

        return "Resolution" + str({'lorentzian': self.e_res})
