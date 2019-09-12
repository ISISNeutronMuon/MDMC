"""Module containing an abstract base class for MD engine facades"""

from abc import ABC, abstractmethod

class MDEngine(ABC):

    """
    Abstract base class for MD engine facades
    """

    @property
    @abstractmethod
    def saved_config(self):

        """
        Get the saved configuration of the atomic positions

        Returns
        -------
        Configuration
            the atomic positions
        """

        pass

    @abstractmethod
    def setup_universe(self, universe, **settings):

        """
        Creates a universe configuration and populates with structural units

        Parameters
        ----------
        universe : Universe
            a molecular dynamics Universe which will be setup in the engine.
        **settings
            The majority of these are generic but some are specific to the
            MDEngine that is being used.
        """

        pass

    @abstractmethod
    def setup_simulation(self, **settings):

        """
        Sets the options required to perform a simulation on a setup universe.
        Must follow a call to setup_universe().

        Parameters
        ----------
        universe : Universe
            a molecular dynamics Universe which will be simulated in the engine.
        settings**
            The majority of these are generic but some are specific to the
            MDEngine that is being used.
        """

        pass

    @abstractmethod
    def minimize(self, n_steps, **settings):

        """
        Minimizes the simulation energy

        Parameters
        ----------
        n_steps : int
            maximum number of steps for the energy minimization.
        """

        pass

    @abstractmethod
    def run(self, n_steps, equilibration):

        """
        Runs a simulation.  Must follow a call to setup_universe() and
        setup_simulation().

        Parameters
        ----------
        n_steps : int
            number of steps for the time integrator.
        equilibration : bool
            If True, run is equilibration which does not store the trajectory.
            Otherwise run is prodution.
        """

        pass

    @abstractmethod
    def convert_trajectory(self):

        """
        Parses the trajectory from the MDEngine format into MDMC format

        Returns
        -------
        Trajectory
            the trajectory from the most recent production simulation
        """

        # convert_trajectory has no range function as it is assumed that the
        # trajectory that is calculated by MD is exactly what is required

        pass

    @abstractmethod
    def update_parameters(self):

        """
        Updates the MD engine force field parameters from the universe
        """

        pass

    @abstractmethod
    def save_config(self):

        """
        Sets self.saved_config to the current configuration
        """

    @abstractmethod
    def reset_config(self):

        """
        Resets the configuration of the simulation to that in saved_config
        """

        pass
