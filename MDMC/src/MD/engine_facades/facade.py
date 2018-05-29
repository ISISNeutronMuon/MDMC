"""Abstract base class for MD engine facades

AUTHOR :    Thomas Farmer        START DATE :    2018-5-16 14:48:12"""

from abc import ABCMeta, abstractmethod

class MDEngine:

    __metaclass__ = ABCMeta

    @abstractmethod
    def setup_universe(self, universe, **settings):
        pass

    @abstractmethod
    def setup_simulation(self, universe, **settings):
        pass

    @abstractmethod
    def run(self, n_steps):
        pass
