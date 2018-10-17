"""Abstract base class for MD engine facades

AUTHOR :    Thomas Farmer        START DATE :    2018-5-16 14:48:12"""

from abc import ABCMeta, abstractmethod

class MDEngine:

    __metaclass__ = ABCMeta

    @abstractmethod
    def setup_universe(self, universe, **settings):

        """
        Creates a universe configuration on which a simulation can be run
        """

        pass

    @abstractmethod
    def setup_simulation(self, universe, **settings):

        """
        Sets the options required to perform a simulation on a setup universe.
        Must follow a call to setup_universe().
        """

        pass

    @abstractmethod
    def minimize(self, n_steps):

        """
        Minimizes the simulation energy

        Arguments:
        n_steps - integer maximum number of steps for minimizer
        """

        pass

    @abstractmethod
    def run(self, n_steps, equilibration):

        """
        Runs a simulation.  Must follow a call to setup_universe() and
        setup_simulation().

        Arguments:
        n_steps - integer number of steps for integrator
        equilibration - Boolean which defines if run is equilibration or
        production
        """

        pass

    @abstractmethod
    def convert_trajectory(self):

        """
        Parses the trajectory from the MDEngine format into MDMC format
        """

        # convert_trajectory has no range function as it is assumed that the
        # trajectory that is calculated by MD is exactly what is required

        pass

    @abstractmethod
    def update_parameters(self):

        """
        Updates the MD engine force field parameters
        """

        pass
