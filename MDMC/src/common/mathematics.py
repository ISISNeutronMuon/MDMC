"""A module containing mathematical functions

AUTHOR :    Thomas Farmer        START DATE :    18/07/2018, 16:42:10"""

import numpy as np
from numpy.fft import fft, ifft

def correlation(input1, input2=None, normalise=False):

    """
    The correlation of two vector

    The Fast Correlation Algorithm (FCA) is utilised.  If only a single input is
    provided, the autocorrelation is calculated.
    """

    N = len(input1)

    fft1 = fft(input1, n=(N * 2), axis=0)

    if input2 is None:
        fft2 = fft1
    else:
        fft2 = fft(input2, n=(N * 2), axis=0)

    # Calculate the cyclic correlation function
    cyclic_corr = ifft(np.conjugate(fft1) * fft2, axis=0)

    # Normalise for variable number of contributions to each timestep:
    # 1 / (N - m)
    # where m is the number of each individual step
    if normalise:
        prefactor = 1. / (N - np.arange(N))
        if len(np.shape(cyclic_corr)) > 1:
            cyclic_corr = np.sum(cyclic_corr, axis=1)
    else:
        prefactor = 1.

    corr = prefactor * np.real(cyclic_corr[0:N])

    return corr

def convolution(input1, input2):

    """
    The convolution of two inputs
    """

    raise NotImplementedError
