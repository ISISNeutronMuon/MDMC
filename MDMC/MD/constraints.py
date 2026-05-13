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

"""Constraint solvers."""


class ConstraintAlgorithm:
    """
    Class describing the algorithm and parameters which are applied to
    constrain ``BondedInteraction`` objects

    Parameters
    ----------
    accuracy : float
        The accuracy (tolerance) of the applied constraints
    max_iterations : int
        The maximum number of iterations that can be used when calculating the
        additional force that is required to constrain the atoms to satisfy the
        constraints on the bonded interactions

    Attributes
    ----------
    accuracy : float
        The accuracy (tolerance) of the applied constraints
    """

    def __init__(self, accuracy: float, max_iterations: int):

        self.accuracy = accuracy
        self.max_iterations = max_iterations

    @property
    def name(self) -> str:
        """
        Get the name of the class

        Returns
        -------
        str
            The name of the class
        """

        return self.__class__.__name__

    @property
    def max_iterations(self) -> int:
        """
        Get or set the maximum number of iterations that can be used when
        calculating the additional force that is required to constrain the atoms
        to satisfy the constraints on the bonded interactions

        Returns
        -------
        int
            The maximum number of iterations
        """

        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, value: int) -> None:

        self._max_iterations = int(value)


class Shake(ConstraintAlgorithm):
    """
    Holds the parameters which are required for the SHAKE algorithm to be
    applied to the constrained interactions

    Parameters
    ----------
    accuracy : float
        The accuracy (tolerance) of the applied constraints
    max_iterations : int
        The maximum number of iterations that can be used when calculating the
        additional force that is required to constrain the atoms to satisfy the
        constraints on the bonded interactions
    """


class Rattle(ConstraintAlgorithm):
    """
    Holds the parameters which are required for the RATTLE algorithm to be
    applied to the constrained interactions

    Parameters
    ----------
    accuracy : float
        The accuracy (tolerance) of the applied constraints
    max_iterations : int
        The maximum number of iterations that can be used when calculating the
        additional force that is required to constrain the atoms to satisfy the
        constraints on the bonded interactions
    """
