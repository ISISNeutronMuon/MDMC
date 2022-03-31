"""Module which defines physical constants"""

from numpy import pi

from MDMC.common.unit_registry import UREG

h = 4.135667696e-15 * UREG.eV * UREG.s
h_bar = h / (2 * pi)
