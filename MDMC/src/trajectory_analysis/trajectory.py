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

    @property
    def atom_positions(self):
        return self.data['position']

    @property
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

    # TODO: Frames currently start at 1, which leads to an offset to the array numbering - do
    # we want to start from frame 0 or alternatively return item-1 from __getitem__()? Although
    # this needs to throw an exception when being called with 0, rather than returning the last
    # item i.e. __getitem__(-1)

    def __init__(self, *configurations):
        self.n_frames = len(configurations)
        self.data = configurations

    @property
    def data(self):
        return self._data

    # TODO: Remove DRY violation
    @data.setter
    def data(self, configurations):
        try:
            self._data = np.array(
                [(i, config.time, config) for i, config in enumerate(configurations, 1)],
                dtype = [('frame', 'int64'),
                ('time', 'float64'),
                ('configuration', 'object')])
        except AttributeError:
            self._data = np.array(
                [(i, config) for i, config in enumerate(configurations, 1)],
                dtype = [('frame', 'int64'),
                ('configuration', 'object')])

    def __getitem__(self, item):

        """
        Indexing and slicing is relative to frames, so is normalised -1 i.e.
        __getitem__(1) returns the configuration of frame 1.
        """

        # TODO: Currently raises TypeError if item is int and < 1 instead of IndexError
        if item > 0 or item.start > 0:
            try:
                return Trajectory(self.data[item-1]['configuration'])
            except TypeError:
                slce = slice(item.start-1,item.stop-1,item.step)
                return Trajectory(*self.data[slce]['configuration'])
        else:
            raise IndexError("Frame indexes start at 1")

    # TODO: Add methods for returning configs for specific frames/times
    @property
    def frames(self):
        return self.data['frame']

    @property
    def times(self):
        return self.data['time']

    @property
    def atoms(self):
        # TODO: Test that all configurations have the same atoms
        """
        Assumes that all configurations have the same atoms
        """

        return self.data['configuration'][0].atom_list

    @property
    def configurations(self):
        return self.data['configuration']

    @property
    def positions(self):
        return np.array([config.atom_positions
            for config in self.configurations])

    @property
    def velocities(self):
        return np.array([config.atom_velocities
            for config in self.configurations])


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
