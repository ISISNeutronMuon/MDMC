"""The Gaussian resolution subclass"""
import numpy as np

from MDMC.resolution.resolution import Resolution
from MDMC.common.constants import h_bar
from MDMC.common.resolution_functions import gaussian


class GaussianResolution(Resolution):
    """
    A `Resolution` subclass for applying a Gaussian resolution
    """

    def __init__(self, e_res):
        # converts energy resolution from user-friendly ueV to system unit meV
        self.e_res = e_res / 1000

    def apply(self, FQt, t, Q):
        N_Q, N_T = np.shape(FQt)
        window = self.window_in_t(t[:N_T])

        return np.broadcast_to(window, (N_Q, N_T)) * FQt

    def window_in_w(self, w, norm=True):
        """
        The Gaussian window in frequency space

        Parameters
        ----------
        w: frequency
        mu: the offset of the function (defaults to 0)
        norm: if True, normalises the distribution to unity.

        Returns
        -------
        The window function in frequency space (i.e. the Gaussian with
        FWHM self.e_res, centred on 0)
        """

        window = gaussian(w, self.e_res, norm)

        return window

    def window_in_t(self, t):
        """
        The Gaussian window in time space

        Parameters
        ----------
        t: time

        Returns
        -------
        The window function in time space (i.e. the Gaussian with FWHM sigma_t, centred on 0)
        """

        # We convert the FWHM energy resolution (in meV) into sigma_t (in fs) using the inverse
        # relationship between the width of a Gaussian and its Fourier transform,
        # rather than explicitly transforming it, then applying a factor
        # of 1e18 to convert from h / h_bar's units of eV s into system units.

        sigma_t = (2 * np.sqrt(2 * np.log(2)) * h_bar.magnitude * 1e18) / self.e_res
        window = gaussian(t, sigma_t, norm=False)

        return window

    def __repr__(self):
        """
        Resolution objects are represented with the dictionary used to create them
        """

        return "Resolution" + str({'gaussian': self.e_res})
