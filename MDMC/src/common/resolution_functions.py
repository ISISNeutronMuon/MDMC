"""A module for containing all resolution functions

AUTHOR :    Thomas Farmer        START DATE :    27/07/2018, 13:40:34"""

from scipy import signal


def Gaussian(N, sigma):

    """
    Returns:
    A Gaussian distribution

    Arguments:
    N: Length of the distribution
    sigma: The standard deviation of the Gaussian
    """

    return signal.gaussian(N, sigma)


def Lorentzian(N, gamma):

    """
    Returns:
    A Lorentzian (Cauchy) distribution

    Arguments:
    N: Length of the distribution
    gamma: half-width at half-maximum
    """

    raise NotImplementedError
