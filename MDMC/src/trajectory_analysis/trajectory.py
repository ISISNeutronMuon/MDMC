"""Module for trajectory and histogram containers

AUTHOR :    Thomas Farmer        START DATE :    2018-5-29 23:44:11"""

import numpy as np

# TODO: Remove this dependency
from MDMC.src.MD.structural_units import Molecule


# TODO: Deal with molecule list better - separate Atomic Configuration?
class Configuration(object):

    """
    A Configuration stores atoms and their positions and velocities (in the form
    of AtomConfig)
    """

    # TODO: Consider how wraparound for periodic objects will work

    def __init__(self, *structural_units):
        self.data = self.create_config_array(*structural_units)
        self.molecule_list = self.create_molecule_list(*structural_units)

    @property
    def atom_list(self):
        return self.data['atom']

    def atom_positions(self):
        return self.data['position']

    def atom_velocities(self):
        return self.data['velocity']

    def create_config_array(self, *structural_units):
        return np.array([(atom, atom.position, atom.velocity)
            for unit in structural_units
            for atom in unit.atom_list],
            dtype=[('atom','object'),
            ('position','object'),
            ('velocity','object')])

    # TODO: change to np.array
    def create_molecule_list(self, *structural_units):
        return [unit for unit in structural_units if isinstance(unit, Molecule)]

    def add_structural_units(self, *structural_units):
        self.data = np.append(
            self.data,self.create_config_array(*structural_units))
        self.molecule_list += (self.create_molecule_list(*structural_units))

    # TODO: Create delete method
    def remove_structural_units(self, *structural_units):
        pass

    # TODO: Add __getitem__ to return atom_list, positions and velocities
    def __getitem__(self, item):
        pass

class TimedConfiguration(Configuration):

    def __init__(self, time, *structural_units):
        super(TimedConfiguration, self).__init__(*structural_units)
        self.time = time


class Trajectory(object):

    """
    A Trajectory is a collection of Configurations

    For calculating dynamic observables, all Configurations must be
    TimedConfigurations
    """

    def __init__(self, *configurations):
        self.n_frames = len(configurations)
        self.data = configurations

    @property
    def data(self):
        return self._data

    # TODO: Remove DRY violation
    @data.setter
    def data(self, configurations):
        if [isinstance(config,TimedConfiguration) for config in configurations]:
            self._data = np.array(
                [(i, config.time, config) for i, config in enumerate(configurations, 1)],
                dtype = [('frame', 'int64'),
                ('time', 'float64'),
                ('configuration', 'object')])
        else:
            self._data = np.array(
                [(i, config) for i, config in enumerate(configurations, 1)],
                dtype = [('frame', 'int64'),
                ('configuration', 'object')])

    def __getitem__(self, item):
        return Trajectory(self.data[item])

    # TODO: Add methods for returning frames, times, configs, as well as configs for specific frames/times


class Histogram(object):

    """
    A Histogram is a rebinned Configuration

    Assumes isotropic configuration
    """

    def __init__(self, trajectory, bin_axis, *bin_axes):
        pass#self.data = configuration

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, configuraton):
        self._data = np.histogram(configuraton)

    # TODO: Consider if this will be extracted into a vector class
    def distance(self,vec1,vec2):
        return np.linalg.norm(vec1-vec2)
