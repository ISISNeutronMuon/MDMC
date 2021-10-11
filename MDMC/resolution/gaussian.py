from MDMC.resolution.resolution import Resolution
from MDMC.common.constants import h_bar
import numpy as np


class GaussianResolution(Resolution):
    """
    A `Resolution` subclass for applying a Gaussian resolution
    """

    # *ignored just takes parameters needed by other resolutions but not this one and ignores them
    def __init__(self, e_res, *ignored):
        self.use_FQT = True
        # converts energy resolution from user-friendly ueV to system unit meV
        self.e_res = e_res / 1000

    def apply(self, fqt, t):
        N_Q, N_T = np.shape(fqt)
        window = self.window_in_t(t[:N_T])

        return np.broadcast_to(window, (N_Q, N_T)) * fqt

    def window_in_w(self, w, mu=0.0, norm=True):
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

        window = np.exp(-0.5 * ((w - mu) / self.e_res)**2)
        if norm:
            window /= (self.e_res * np.sqrt(2.0 * np.pi))

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

        sigma_t = (2 * np.sqrt(2 * np.log(2)) * h_bar * 1e18) / self.e_res
        window = np.exp(-0.5 * (t / sigma_t)**2)

        return window

    def __repr__(self):
        """
        Resolution objects are represented with the dictionary used to create them
        """

        return "Resolution" + str({'gaussian': self.e_res})
