"""Module containing an abstract base class for MD engine facades"""
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from MDMC.trajectory_analysis.trajectory import Configuration
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory

if TYPE_CHECKING:
    from MDMC.MD.simulation import Simulation

class MDEngine(ABC):

    """
    Abstract base class for MD engine facades
    """

    def __repr__(self) -> str:

        return ('<{0}\n'
                ' {{MD_engine: {MD_engine},\n'
                ' exp_datasets: {exp_datasets},\n'
                ' FoM_calculator: {FoM_calculator},\n'
                ' minimizer: {minimizer},\n'
                ' reset_config: {reset_config},\n'
                ' fit_parameters: {fit_parameters},\n'
                ' settings: {settings}}}>').format(self.__class__.__name__,
                                                   **self.__dict__)

    @property
    @abstractmethod
    def saved_config(self) -> Configuration:
        """
        Get the saved configuration of the atomic positions

        Returns
        -------
        ``Configuration``
            The atomic positions
        """

        raise NotImplementedError

    @property
    def parent_simulation(self) -> 'Simulation':
        """
        Get or set the simulation that created this engine facade

        Returns
        -------
        `Simulation`
            The Simulation object that created this engine facade
        """

        try:
            return self._parent_simulation
        except AttributeError as error:
            raise AttributeError("This MD engine does not belong to a simulation. "
                                 "MD engines should be created through initialising Simulations."
                                 "") from error

    @parent_simulation.setter
    def parent_simulation(self, value: 'Simulation') -> None:
        # pylint: disable=attribute-defined-outside-init
        # as this is internal and abstract
        self._parent_simulation = value

    @property
    def time_step(self) -> float:
        """
        Get the simulation time step in ``fs`` from the parent simulation

        Returns
        -------
        `float`
            Simulation time step in ``fs``
        """

        return self.parent_simulation.time_step

    @property
    def traj_step(self) -> int:
        """
        Get the number of simulation steps between saving the
        ``CompactTrajectory`` from the parent simulation

        Returns
        -------
        `int`
            Number of simulation steps that elapse between the
            ``CompactTrajectory`` being stored
        """

        try:
            return self.parent_simulation.traj_step
        except AttributeError:
            return None

    @abstractmethod
    def setup_universe(self, universe: str, **settings: dict) -> None:
        """
        Creates a ``Universe.configuration`` and populates with
        ``Structure``

        Parameters
        ----------
        universe : Universe
            A molecular dynamics ``Universe`` which will be setup in the
            ``MDEngine``.
        **settings
            The majority of these are generic but some are specific to the
            ``MDEngine`` that is being used.
        """

        raise NotImplementedError

    @abstractmethod
    def setup_simulation(self, **settings: dict) -> None:
        """
        Sets the options required to perform a simulation on a setup
        ``Universe``. Must follow a call to ``setup_universe()``.

        Parameters
        ----------
        universe : Universe
            A molecular dynamics ``Universe`` which will be simulated in the
            ``MDEngine``.
        settings**
            The majority of these are generic but some are specific to the
            ``MDEngine`` that is being used.
        """

        raise NotImplementedError

    @abstractmethod
    def minimize(self, n_steps: int, minimize_every: int = 10,
                 **settings: dict) -> None:
        """
        Minimizes the simulation energy

        Parameters
        ----------
        n_steps : int
            Maximum number of MD steps during the energy minimization.
        minimize_every : int
            Number of MD steps between two consecutive minimizations.
        """

        raise NotImplementedError

    @abstractmethod
    def run(self, n_steps: int, equilibration: bool) -> None:
        """
        Runs a simulation.  Must follow a call to ``setup_universe()`` and
        ``setup_simulation()``.

        Parameters
        ----------
        n_steps : int
            Number of steps for the time integrator.
        equilibration : bool
            If `True`, run is equilibration which does not store the
            ``trajectory``. Otherwise run is prodution.
        """

        raise NotImplementedError

    @abstractmethod
    def convert_trajectory(self, start: int = 0, stop: int = None,
                           step: int = 1, **settings: dict) -> CompactTrajectory:
        """
        Parses the trajectory from the ``MDEngine`` format into MDMC format

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

        # convert_trajectory has no range function as it is assumed that the
        # trajectory that is calculated by MD is exactly what is required

        raise NotImplementedError

    @abstractmethod
    def update_parameters(self) -> None:
        """
        Updates the ``MDEngine`` force field ``Parameter`` objects from the
        ``Universe``
        """

        raise NotImplementedError

    @abstractmethod
    def save_config(self) -> None:
        """
        Sets ``self.saved_config`` to the current configuration
        """

        raise NotImplementedError

    @abstractmethod
    def reset_config(self) -> None:
        """
        Resets the configuration of the simulation to that in ``saved_config``
        """

        raise NotImplementedError
