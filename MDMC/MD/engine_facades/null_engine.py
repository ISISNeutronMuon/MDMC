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

"""Module containing an MD engine that does not run any MD simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from MDMC.MD.engine_facades.facade import MDEngine
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory
from MDMC.trajectory_analysis.trajectory import Configuration

if TYPE_CHECKING:
    from MDMC.MD.simulation import Universe


class NullEngine(MDEngine):
    """An MD engine which does not run any MD.

    This is meant to test the optimisation procedure by generating results quickly.
    It passes the parameters from the Universe object to the observable calculation
    by setting attributes of the output trajectory to be parameter values."""

    @property
    def saved_config(self) -> Configuration:
        """
        Returns an empty configuration.

        Returns
        -------
        ``Configuration``
            The atomic positions
        """
        return Configuration()

    def setup_universe(self, universe: Universe, **settings: Any) -> None:
        """
        Copies the parameters from the Universe object.

        Parameters
        ----------
        universe : Universe
            A molecular dynamics ``Universe`` which will be setup in the
            ``MDEngine``.
        **settings
            The majority of these are generic but some are specific to the
            ``MDEngine`` that is being used.
        """
        self.parameters = universe.parameters

    def setup_simulation(self, **settings: Any) -> None:
        """
        Do nothing.

        Parameters
        ----------
        universe : Universe
            A molecular dynamics ``Universe`` which will be simulated in the
            ``MDEngine``.
        settings**
            The majority of these are generic but some are specific to the
            ``MDEngine`` that is being used.
        """
        pass

    def minimize(self, n_steps: int, minimize_every: int = 10, **settings: Any) -> None:
        """
        Do nothing.

        Parameters
        ----------
        n_steps : int
            Maximum number of MD steps during the energy minimization.
        minimize_every : int, optional, default 10
            Number of MD steps between two consecutive minimizations.
        """
        pass

    def run(self, n_steps: int, equilibration: bool, **kwargs) -> None:
        """
        Do nothing.

        Parameters
        ----------
        n_steps : int
            Number of steps for the time integrator.
        equilibration : bool
            If `True`, run is equilibration which does not store the
            ``trajectory``. Otherwise run is prodution.
        """
        pass

    def convert_trajectory(
        self,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
        **settings: Any,
    ) -> CompactTrajectory:
        """
        Return an empty trajectory.

        Parameters are saved as attributes of the trajectory instance.

        Parameters
        ----------
        start : int
            The index of the first trajectory, inclusive.
        stop : int
            The index of the last trajectory, exclusive.
        step : int
            The step size between trajectories.
        **settings
            N/A

        Returns
        -------
        ``CompactTrajectory``
            The ``CompactTrajectory`` from the most recent production simulation
        """
        temp = CompactTrajectory(n_steps=1)
        temp.parameters = {par.name.split()[0]: par.value for par in self.parameters.values()}
        return temp

    def update_parameters(self) -> None:
        """
        Do nothing.
        """
        pass

    def save_config(self) -> None:
        """
        Do nothing.
        """
        pass

    def clear(self) -> None:
        """
        Do nothing.
        """
        pass

    def reset_config(self) -> None:
        """
        Do nothing.
        """
        pass

    def eval(self, variable: str) -> Any:
        """
        Return 1.0
        """
        return 1.0
