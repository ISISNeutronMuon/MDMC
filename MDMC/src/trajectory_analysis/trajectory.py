"""Module for trajectory and histogram containers

AUTHOR :    Thomas Farmer        START DATE :    2018-5-29 23:44:11"""

import numpy as np

from MDMC.src.MD.structural_units import Molecule


# TODO: Deal with molecule list better - separate Atomic Configuration?
class Configuration(object):

    """
    A Configuration stores atoms and their positions and velocities (in the form
    of AtomConfig)
    """

    # TODO: Consider how wraparound for periodic objects will work

    def __init__(self, *structural_units):
        self.data = self.create_config_dict(*structural_units)
        self.molecule_list = self.create_molecule_list(*structural_units)

    @property
    def atom_list(self):
        return self.data.keys()

    def create_config_dict(self, *structural_units):
        return {atom:AtomConfig(atom)
            for unit in structural_units
            for atom in unit.atom_list}

    def create_molecule_list(self, *structural_units):
        return [unit for unit in structural_units if isinstance(unit, Molecule)]

    # TODO: Create add and delete atom methods
    def add_structural_units(self, *structural_units):
        self.data.update(self.create_config_dict(*structural_units))
        self.molecule_list += (self.create_molecule_list(*structural_units))

    # TODO: Create delete method
    def remove_structural_units(self, *structural_units):
        pass


class TimedConfiguration(Configuration):

    def __init__(self, time, *structural_units):
        super(TimedConfiguration, self).__init__(structural_units)
        self.time = time


class AtomConfig(object):

    def __init__(self, atom):
        self.position = atom.position
        self.velocity = atom.velocity


class Trajectory(object):

    """
    A Trajectory is a collection of Configurations

    For calculating dynamic observables, all Configurations must be
    TimedConfigurations
    """

    def __init__(self, *configurations):
        self.data = []

    # TODO: Define __getitem
    def __getitem__(self, item):
        return Trajectory(self.data[item])


class Histogram(object):

    """
    A Histogram is a rebinned Configuration
    """

    def __init__(self, trajectory, bin_axis, *bin_axes):
        pass
        self.data = configuration

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, configuraton):
        self._data = np.histogram(configuraton)


class HistogramCollection(object):

    """
    A collection of histograms created from a Trajectory
    """

    def __init__(self, trajectory):
        self.data = trajectory

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, trajectory):
        self._data = [Histogram(config) for config in trajectory]
