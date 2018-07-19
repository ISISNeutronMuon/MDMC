"""A module containing mathematical functions

AUTHOR :    Thomas Farmer        START DATE :    18/07/2018, 16:42:10"""

from numpy import conjugate, arange, real
from numpy.fft import fft, ifft

def correlation(input1, input2=None, normalise=False):

    """
    The correlation of two vector

    The Fast Correlation Algorithm (FCA) is utilised.  If only a single input is
    provided, the autocorrelation is calculated.
    """

    N = len(input1)

    fft1 = fft(input1, n=(N * 2))

    if input2 is None:
        fft2 = fft1
    else:
        fft2 = fft(input2, n=(N * 2))

    # Calculate the cyclic correlation function
    cyclic_corr = ifft(conjugate(fft1) * fft2)

    # Normalise for variable number of contributions to each timestep:
    # 1 / (N - m)
    # where m is the number of each individual step
    if normalise:
        prefactor = 1. / (N - arange(N))
    else:
        prefactor = 1.

    corr = prefactor * real(cyclic_corr[0:N])

    return corr

def convolution(input1, input2):

    """
    The convolution of two inputs
    """

    raise NotImplementedError
