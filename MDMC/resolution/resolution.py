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

"""The Resolution abstract base class."""
from abc import ABC, abstractmethod

import numpy as np


class Resolution(ABC):
    """
    An abstract base class for resolution functions.
    """
    # pylint: disable=too-few-public-methods

    @abstractmethod
    def apply(self, FQt: np.ndarray, t: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """
        Apply resolution to an FQt array.

        Parameters
        ----------
        FQt : ~numpy.ndarray
            the FQt array to which resolution is applied.
        t : ~numpy.ndarray
            the time points for the FQt array.
        Q : ~numpy.ndarray
            the momentum points for the FQt array.

        Returns
        -------
        ~numpy.ndarray
            The array with the resolution function applied to it.
        """

        raise NotImplementedError
