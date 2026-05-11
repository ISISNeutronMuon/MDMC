# MDMC is a package for the optimisation of classical potentials with experimental data
# Copyright (C) 2026 MDMC Developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
        A discretised Gaussian distribution of the same length as ``x``.
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
        A discretised Lorentzian distribution of the same length as ``x``.
    """

    y = (1 / np.pi) * ((0.5 * gamma) / ((x - x_0) ** 2 + (0.5 * gamma) ** 2))

    return y
