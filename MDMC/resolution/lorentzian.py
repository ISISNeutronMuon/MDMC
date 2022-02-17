"""The Lorentzian resolution subclass"""
import numpy as np

from MDMC.resolution.resolution import Resolution
from MDMC.common.resolution_functions import lorentzian


class LorentzianResolution(Resolution):
    """
    A `Resolution` subclass for applying a Lorentzian resolution
    """

    def __init__(self, e_res):
        self.e_res = e_res / 1000

    def apply(self, FQt, t, Q):
        N_Q, N_T = np.shape(FQt)
        window = self.window_in_t(t[:N_T])

        return np.broadcast_to(window, (N_Q, N_T)) * FQt

    def window_in_w(self, w):
        """
        The Lorentzian window in frequency space
        """

        window = lorentzian(w, self.e_res)

        return window

    def window_in_t(self, t):
        """
        The Lorentzian window in time space (i.e. the Fourier transform centred around zero)
        The Fourier transform of the Lorentzian is
        F(k) = e^((2 * pi * i) * k * x_0) - Gamma * pi * |k|
        where x_0 is the offset and Gamma the FWHM.
        thus as the instrument resolution function is centred around x_0, this simplifies to
        e^(-Gamma * pi * |k|).
        """

        window = np.exp((-self.e_res * np.pi * np.abs(t)))

        return window

    def __repr__(self):
        """
        Resolution objects are represented with the dictionary used to create them
        """

        return "Resolution" + str({'lorentzian': self.e_res})
