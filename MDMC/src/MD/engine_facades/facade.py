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
    def run(self, n_steps):

        """
        Runs a simulation.  Must follow a call to setup_universe() and
        setup_simulation().
        """

        pass
