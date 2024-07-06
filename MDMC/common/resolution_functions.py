"""
A module for containing all resolution functions.
"""

import numpy as np


def gaussian(x: np.ndarray, sigma: float, mu: float = 0.0, norm: bool = True) -> np.ndarray:
    """
    Calculate the Gaussian distribution.

    Parameters
    ----------
    x : numpy.ndarray
        The x values at which the Gaussian distribution is calculated.
    sigma : float
        The standard deviation of the Gaussian.
    mu : float, optional
        The offset of the Gaussian.
    norm : bool, optional
        If `True`, resulting distribution is normalized to unity.

    Returns
    -------
    numpy.ndarray
        An ``array`` with the same length as ``x``, with a Gaussian distribution.
    """

    y = np.exp(-0.5 * ((x - mu) / sigma)**2)

    if norm:

        y /= (sigma * np.sqrt(2.0 * np.pi))

    return y


def lorentzian(x: np.ndarray, gamma: float, x_0: float = 0.0) -> np.ndarray:
    """
    Calculate the Lorentzian (Cauchy) distribution.

    Parameters
    ----------
    x : numpy.ndarray
        The x values at which the Gaussian distribution is calculated.
    gamma : float
        The full-width at half-maximum.
    x_0 : float, optional
        The offset of the distribution.

    Returns
    -------
    numpy.ndarray
        An ``array`` with the same length as ``x``, with the Lorentzian
        distribution.
    """

    y = (1 / np.pi) * ((0.5 * gamma) / ((x - x_0) ** 2 + (0.5 * gamma) ** 2))

    return y
