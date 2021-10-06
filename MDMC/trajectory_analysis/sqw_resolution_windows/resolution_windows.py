"""A module for resolution window functions."""
import MDMC.common.resolution_functions
from MDMC.common.constants import h_bar
import numpy as np


def gaussian_window(t: np.ndarray, fwhm):

    """
    Calculates the Gaussian instrument resolution window in FQt space

    Parameters
    ----------
    t: the range of t-values for the FQt that resolution will be applied to.
    fwhm: the full-width half-maximum for your instrument resolution.

    Returns
    -------
    window: the window function to be multiplied with FQt to apply resolution.
    """

    sigma_t = (2 * np.sqrt(2 * np.log(2)) * h_bar * 1e18) / fwhm
    window = MDMC.common.resolution_functions.gaussian(t, sigma_t, norm=False)

    return window


def lorentzian_window(t: np.ndarray, fwhm):

    """
    Calculates the Lorentzian instrument resolution window in FQt space

    Parameters
    ----------
    t: the range of t-values for the FQt that resolution will be applied to.
    fwhm: the full-width half-maximum for your instrument resolution.

    Returns
    -------
    window: the window function to be multiplied with FQt to apply resolution.
    """

    window = np.exp((-fwhm * np.pi * np.abs(t)))

    return window

