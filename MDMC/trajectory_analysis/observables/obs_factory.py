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
Factory class for generating observables.
"""

from collections.abc import Callable

from MDMC.common.factory import RegisterFactory
from MDMC.trajectory_analysis.observables.obs import Observable


class ObservableFactory(RegisterFactory[Observable]):
    """
    Provide a factory for creating an :class:`Observable`.

    Any module within the observables submodule can be created with a
    string of the class name, as long as it is a subclass of
    ``Observable``.
    """

    registry: dict[str, Observable] = {}

    @classmethod
    def create(cls, key: str, *args, **kwargs) -> Callable[..., Observable]:
        """
        Return an instance of given class.
        """
        obs = super().create(key, *args, **kwargs)
        obs.name = key
        return obs
