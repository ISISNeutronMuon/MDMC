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
    # TODO: Consider how filtering by element will work - should config array include element type?

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

    @classmethod
    def add_configurations(cls, *configurations):

        """
        Returns a new configuration from the sum of configurations
        """

        # TODO: Currently doesn't add molecule lists - add this but consider if a molecule set would be preferable
        structural_units = [atom for config in configurations
            for atom in config.data['atom']]
        return cls(*structural_units)

    def __len__(self):
        return len(self.atom_list)

class TemporalConfiguration(Configuration):

    def __init__(self, time, *structural_units):
        super(TemporalConfiguration, self).__init__(*structural_units)
        self.time = time

    @classmethod
    def add_configurations(cls, *configurations, **set_time):

        # TODO: Add warning if configurations don't all have the same time
        time = set_time.get('time',
            np.mean([config.time for config in configurations]))

        # TODO: Currently doesn't support molecule lists - add this
        structural_units = [atom for config in configurations
            for atom in config.data['atom']]
        return cls(time, *structural_units)


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
        Indexing and slicing is relative to time
        """

        # TODO: Change filter_configs_by_time so that a single time can also be passed

        try:
            return Trajectory(*self.filter_configs_by_time(
                item.start,item.stop))
        except AttributeError:
            raise ValueError("Trajectory can only be sliced, not indexed")

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
        return np.array([position for config in self.configurations
            for position in config.atom_positions])

    @property
    def velocities(self):
        return np.array([velocity for config in self.configurations
            for velocity in config.atom_velocities])

    def filter_configs_by_time(self, start, end):

        """
        Returns configurations with times in half open interval defined by start
        and end
        """

        return self.configurations[(self.times >= start) & (self.times < end)]

    def __len__(self):
        return sum([len(config) for config in self.data['configuration']])


class Histogram(object):

    """
    A Histogram is a rebinned Trajectory

    Assumes isotropic configuration
    """

    # TODO: Implement logic for velocity as well as position
    def __init__(self, trajectory, **axes_bins):

        self.trajectory = trajectory

        self.r_axis = axes_bins['r']
        self.time_axis = axes_bins.get('time', None)
        self.distance_data = np.array([(frame['time'],
            list(self._calculate_distances(
            frame['configuration'].atom_positions)))
            for frame in trajectory.data],
        dtype = [('time', 'float64'), ('distances', 'object')])

        if self.time_axis:
            self._rebin_time()

        self.data = self._histogram_distance_data(self.r_axis)

    # TODO: Consider if this will be extracted into a vector class
    def _distance(self,vec1,vec2):
        return np.linalg.norm(vec1-vec2)

    def _calculate_distances(self, positions):

        """
        Returns a generator of pairwise distances
        """

        # TODO: Something more pythonic
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                yield self._distance(positions[i], positions[j])

    def _calculate_histogram(self, input, axis_bins):

        """
        Returns a histogram with n_bins determined from axis_bins

        All bins are half open to the right, except the final bin which is fully
        open.
        """

        # TODO: Change to make axis values more explicit (as [start,end,step] - maybe change to dict?
        # TODO: Deal with non integer n_bins in a better way
        # TODO: Change so that it can accept a generator input
        n_bins = self._calc_n_bins(axis_bins)
        return np.histogram(input, n_bins, range = (axis_bins[0], axis_bins[1]))

    def _histogram_distance_data(self, axis_bins):

        """
        Returns histograms for all configurations in the distance data
        """

        return np.array(
            [(frame['time'],
            self._calculate_histogram(frame['distances'], axis_bins))
            for frame in self.distance_data],
            dtype = [('time', 'float64'),
            ('histogram', 'object')])

    # TODO: Generalise to _rebin
    def _rebin_time(self):

        """
        Rebins distance data

        Determines the upper and lower bounds and the centers of each bin.
        These are used to rebin the configurations for the trajectory, which is
        then returned. Uses Trajectory __getitem__ to produce a trajectory for
        each time bin. For each of these trajectories the configurations are
        added and then used to return a new trajectory.
        """

        n_bins = self._calc_n_bins(self.r_axis)
        bin_bounds = np.linspace(self.r_axis[0], self.r_axis[1], n_bins+1)
        bin_centers = bin_bounds[0:-1] + (bin_bounds[1] - bin_bounds[0]) / 2.

        trajectories = [trajectory[bin_bounds[i]:bin_bounds[i+1]]
            for i in range(len(bin_centers))]

        # TODO: Fix fudge which sets configuration times = bin_centers
        configurations = [TemporalConfiguration.add_configurations(
            *traj.data['configuration'], time=bin_centers[i])
            for i,traj in enumerate(trajectories)]

        return Trajectory(*configurations)

    def _calc_n_bins(self, axis_bins):
        return int((axis_bins[1] - axis_bins[0]) / axis_bins[2])

    def filter_distance_data_by_time(self, start, end):

        """
        Returns configurations with times in half open interval defined by start
        and end
        """

        return self.distance_data['distances'][
            (self.distance_data['time'] >= start) &
            (self.distance_data['time'] < end)]






# End
