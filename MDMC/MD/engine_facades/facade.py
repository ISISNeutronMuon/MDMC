"""Module containing an abstract base class for MD engine facades"""

from abc import ABCMeta, abstractmethod, abstractproperty

class MDEngine:

    __metaclass__ = ABCMeta

    @abstractproperty
    def saved_config(self):

        """
        A saved configuration of the atomic positions
        """

        pass

    @abstractmethod
    def setup_universe(self, universe, **settings):

        """
        Creates a universe configuration and populates with structural units

        Arguments:
        universe - a Universe object
        settings - The majority of these are generic but some are specific to
        the MDEngine that is being used
        """

        pass

    @abstractmethod
    def setup_simulation(self, **settings):

        """
        Sets the options required to perform a simulation on a setup universe.
        Must follow a call to setup_universe().

        Arguments:
        universe - a Universe object
        settings - The majority of these are generic but some are specific to
        the MDEngine that is being used
        """

        pass

    @abstractmethod
    def minimize(self, n_steps, **settings):

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
