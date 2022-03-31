"""A module for containing all resolution functions"""

import numpy as np


def gaussian(x, sigma, mu=0.0, norm=True):
    """
    Calculates the Gaussian distribution

    Parameters
    ---------
    x : numpy.ndarray
        The x values at which the Gaussian distribution is calculated.
    sigma : float
        The standard deviation of the Gaussian.
    mu : float, optional
        The offset of the Gaussian. Default is 0.0
    norm : bool
        If `True`, resulting distribution is normalized to unity. Default is
        `True`.

    Returns
    -------
    numpy.ndarray
        An ``array`` with the same length as ``x``, with the Gaussian
        distribution
    """

    y = np.exp(-0.5 * ((x - mu) / sigma)**2)

    if norm:

        y /= (sigma * np.sqrt(2.0 * np.pi))

    return y


def lorentzian(x, gamma, x_0=0.0):
    """
    Calculates the Lorentzian (Cauchy) distribution

    Parameters
    ---------
    x : numpy.ndarray
        The x values at which the Gaussian distribution is calculated.
    gamma : float
        The full-width at half-maximum
    x_0 : float
        The offset of the distribution. Defaults to 0.0.

    Returns
    -------
    numpy.ndarray
        An ``array`` with the same length as ``x``, with the Lorentzian
        distribution
    """

    y = (1 / np.pi) * ((0.5 * gamma) / ((x - x_0) ** 2 + (0.5 * gamma) ** 2))

    return y
