"""A module for containing all resolution functions

AUTHOR :    Thomas Farmer        START DATE :    27/07/2018, 13:40:34"""

import numpy as np


def gaussian(x, sigma, mu=0.0, norm=True):

    """
    Arguments:
    x - an an array of floats
    sigma - a float specifying the standard deviation of the Gaussian
    mu - a float specifying the offset
    norm - a boolean specifying if the resulting distribution is normalized
    to unity

    Returns:
    Aan array of floats with the Gaussian distribution
    """

    y = np.exp(-0.5 * ((x - mu) / sigma)**2)

    if norm:

        y /= (sigma * np.sqrt(2.0 * np.pi))

    return y


def lorentzian(N, gamma):

    """
    Returns:
    A Lorentzian (Cauchy) distribution

    Arguments:
    N: Length of the distribution
    gamma: half-width at half-maximum
    """

    raise NotImplementedError
