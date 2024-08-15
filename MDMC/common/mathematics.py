"""
A module containing mathematical functions.
"""

from typing import Union, overload

import numpy as np
from numpy.fft import fft, ifft

# It may still be interesting for futre performance tuning
# to try to replace numpy.fft with pyfftw.
# The easiest way would be to use:
# from pyfftw.interfaces.numpy_fft import fft
# from pyfftw.interfaces.numpy_fft import ifft
# However, probably the multiprocessing module is going
# to be the best solution to improve the performance.

#: Array of standard unit vectors.
#: Used as a standard basis of 3D space.
UNIT_VECTOR = np.array([[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]])

@overload
def correlation(input1: np.ndarray, *, normalise: bool = False) -> np.ndarray: ...


@overload
def correlation(input1: np.ndarray, input2: np.ndarray, normalise: bool = False) -> np.ndarray: ...


def correlation(input1: np.ndarray,
                input2: np.ndarray = None,
                normalise: bool = False) -> np.ndarray:
    """
    Compute the correlation of two vectors.

    The Fast Correlation Algorithm (FCA) is utilised.

    If only a single input is provided, the autocorrelation is calculated.

    Parameters
    ----------
    input1 : numpy.ndarray
        A 1D ``array`` of data.
    input2 :  numpy.ndarray, optional
        A 1D ``array`` of data. If `None`, autocorrelation of ``input1`` is
        calculated. Default is `None`.
    normalise : bool, optional
        If `True`, the correlation is normalised at each point to the number of
        contributions to that point. Default is `False`.

    Returns
    -------
    numpy.ndarray
        A 1D ``array`` of the same length as the ``input1`` containing the
        correlation between ``input1`` and ``input2`` (or autocorrelation of
        ``input1`` if ``input2`` is `None`).
    """

    num_steps = len(input1)

    fft1 = fft(input1, n=(num_steps * 2), axis=0)

    fft2 = (fft1
            if input2 is None else
            fft(input2, n=(num_steps * 2), axis=0))

    # Calculate the cyclic correlation function
    cyclic_corr = ifft(np.conjugate(fft1) * fft2, axis=0)

    # Normalise for variable number of contributions to each correlation:
    # 1 / (N - m)
    # where m is the number of each individual step
    if normalise:
        prefactor = 1. / (num_steps - np.arange(num_steps))
        if len(np.shape(cyclic_corr)) > 1:
            cyclic_corr = np.sum(cyclic_corr, axis=1)
    else:
        prefactor = 1.

    corr = prefactor * np.real(cyclic_corr[0:num_steps])

    return corr


def faster_correlation(input1: np.ndarray, input2: np.ndarray) -> np.ndarray:
    """
    Compute the correlation of two vectors.

    The Fast Correlation Algorithm (FCA) is utilised.

    Parameters
    ----------
    input1 : numpy.ndarray
        A 1D ``array`` of data.
    input2 :  numpy.ndarray, optional
        A 1D ``array`` of data.

    Returns
    -------
    numpy.ndarray
        A 1D ``array`` of the same length as the ``input1`` containing the
        correlation between ``input1`` and ``input2`` (or autocorrelation of
        ``input1`` if ``input2`` is `None`).
    """

    num_steps = len(input1)

    fft1 = fft(input1, n=(num_steps * 2), axis=0)

    fft2 = fft(input2, n=(num_steps * 2), axis=0)

    # Calculate the cyclic correlation function
    cyclic_corr = ifft(np.conjugate(fft1) * fft2, axis=0)

    # Normalise for variable number of contributions to each correlation:
    # 1 / (N - m)
    # where m is the number of each individual step
    prefactor = 1. / (num_steps - np.arange(num_steps))
    # I have to guarantee that the array is a 2D array on input
    cyclic_corr = np.sum(cyclic_corr, axis=1)

    corr = prefactor * np.real(cyclic_corr[0:num_steps])

    return corr


def faster_autocorrelation(input1: np.ndarray,
                           weights: Union[np.ndarray, float] = None) -> np.ndarray:
    """
    The autocorrelation of a vector.

    The Fast Correlation Algorithm (FCA) is utilised.

    Parameters
    ----------
    input1 : numpy.ndarray
        A 1D ``array`` of data.
    weights : np.ndarray or float
        Either weights for each point or single weight for all points.

    Returns
    -------
    numpy.ndarray
        A 1D ``array`` of the same length as the ``input1`` containing the
        correlation between ``input1`` and ``input2`` (or autocorrelation of
        ``input1`` if ``input2`` is `None`).
    """

    num_steps = len(input1)

    fft1 = fft(input1, n=(num_steps * 2), axis=0)

    # Calculate the cyclic correlation function
    cyclic_corr = ifft(np.conjugate(fft1) * fft1, axis=0)

    # Normalise for variable number of contributions to each correlation:
    # 1 / (num_steps - m)
    # where m is the number of each individual step
    prefactor = 1. / (num_steps - np.arange(num_steps))
    # I have to guarantee that the array is a 2D array on input
    if weights is not None:
        try:
            temp_weights = weights.reshape((1, len(weights)))
        except AttributeError:
            temp_weights = weights
        cyclic_corr *= temp_weights
    cyclic_corr = np.sum(cyclic_corr, axis=1)

    corr = prefactor * np.real(cyclic_corr[0:num_steps])

    return corr
