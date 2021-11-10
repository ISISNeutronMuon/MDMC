from MDMC.resolution.resolution import Resolution
from MDMC.common.resolution_functions import lorentzian
import numpy as np


class LorentzianResolution(Resolution):
    """
    A `Resolution` subclass for applying a Lorentzian resolution
    """

    def __init__(self, e_res):
        self.e_res = e_res / 1000

    def _calculate_resolution_window(self, x, frequency_space=False):
        if frequency_space:
            return self._window_in_w(x)
        else:
            return self._window_in_t(x)

    def _window_in_w(self, E):
        """
        The Lorentzian window in frequency space
        """

        window = lorentzian(E, self.e_res)

        return window

    def _window_in_t(self, t):
        """
        The Lorentzian window in time space (i.e. the Fourier transform centred around zero)
        The Fourier transform of the Lorentzian is F(k) = e^((2 * pi * i) * k * x_0) - Gamma * pi * |k|
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
