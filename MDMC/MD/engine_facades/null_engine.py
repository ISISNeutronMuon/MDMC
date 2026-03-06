"""Module containing an abstract base class for MD engine facades"""
from abc import abstractmethod
from typing import Any

from MDMC.MD.engine_facades.facade import MDEngine
from MDMC.MD.simulation import Simulation
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory
from MDMC.trajectory_analysis.trajectory import Configuration


class NullEngine(MDEngine):
    """An MD engine which does not run any MD.

    This is meant to test the optimisation procedure by generating results quickly."""


    @property
    def saved_config(self) -> 'Configuration':
        """
        Get the saved configuration of the atomic positions

        Returns
        -------
        ``Configuration``
            The atomic positions
        """
        return Configuration()

    def setup_universe(self, universe: str, **settings: dict) -> None:
        """
        Do nothing.

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

    def setup_simulation(self, **settings: dict) -> None:
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

    def minimize(self, n_steps: int, minimize_every: int = 10,
                 **settings: dict) -> None:
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

    def convert_trajectory(self, start: int = 0, stop: int = None,
                           step: int = 1, **settings: dict) -> 'CompactTrajectory':
        """
        Return an empty trajectory.

        Parameters
        ----------
        start : int
            The index of the first trajectory, inclusive.
        stop : int
            The index of the last trajectory, exclusive.
        step : int
            The step size between trajectories.
        **settings
            ``scaled_positions`` (`bool`)
                If the ``trajectory_file`` has scaled ``positions``
            ``atom_IDs`` (`list`)
                LAMMPS ``ID`` of the atoms which should be included. If not passed
                then all atoms are included in the converted trajectory.

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
        Updates the ``MDEngine`` force field ``Parameter`` objects from the
        ``Universe``
        """
        pass

    def save_config(self) -> None:
        """
        Sets ``self.saved_config`` to the current configuration
        """
        pass

    def clear(self) -> None:
        """
        Deletes all atoms of the MD engine, restores all settings to their default values,
        and frees all memory.
        """
        pass

    def reset_config(self) -> None:
        """
        Resets the configuration of the simulation to that in ``saved_config``
        """
        pass

    def eval(self, variable: str) -> Any:
        """
        Evaluates some simulation variable.
        """
        return 0.0
