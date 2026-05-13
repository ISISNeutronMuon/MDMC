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

"""k-Space models for long range forces."""


class KSpaceSolver:
    """
    Class describing the k-space solver that is applied to electrostatic and/or
    dispersion interactions

    Different ``MDEngine`` require different parameters to be specified for a
    k-space solver to be used. These parameters are specified in settings.

    Parameters
    ----------
    **settings
        ``accuracy`` (`float`)
            The relative RMS error in per-atom forces

    Attributes
    ----------
    accuracy : float
        The relative RMS error in per-atom forces
    """

    def __init__(self, *, accuracy: float | None = None):

        self.accuracy = accuracy

    @property
    def name(self):
        """
        Get the name of the class

        Returns
        -------
        str
            The name of the class
        """

        return self.__class__.__name__


class Ewald(KSpaceSolver):
    """
    Holds the parameters that are required for the Ewald solver to be applied to
    both/either the electrostatic and/or dispersion interactions

    Parameters
    ----------
    **settings
        ``accuracy`` (`float`)
            The relative RMS error in per-atom forces
    """


class PPPM(KSpaceSolver):
    """
    Holds the parameters that are required for the PPPM solver to be applied to
    both/either the electrostatic and/or dispersion interactions

    Parameters
    ----------
    **settings
        ``accuracy`` (`float`)
            The relative RMS error in per-atom forces
    """

    def __eq__(self, other) -> bool:
        """
        Two KSpaceSolvers are equal if their __dict__ are equal
        """

        if not isinstance(other, self.__class__):
            return False
        return all(v == getattr(other, k) for k, v in self.__dict__.items())

    def __ne__(self, other) -> bool:

        return not self.__eq__(other)
