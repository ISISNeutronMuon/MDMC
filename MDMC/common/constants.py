"""Module which defines physical constants"""

from numpy import pi

#: Planck's constant in eV s
h: float = 4.135667696e-15

#: Reduced Planck's constant in eV s
h_bar: float = h / (2 * pi)
