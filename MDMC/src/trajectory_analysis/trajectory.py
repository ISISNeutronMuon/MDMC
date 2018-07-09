"""Module for trajectory and histogram containers

AUTHOR :    Thomas Farmer        START DATE :    2018-5-29 23:44:11"""

import numpy as np

# TODO: Remove this dependency
from MDMC.src.MD.structural_units import Molecule

# TODO: SORT OUT EFFECTS OF PBC

# TODO: Deal with molecule list better - separate Atomic Configuration?
class Configuration(object):

    """
    A Configuration stores atoms and their positions and velocities (in the form
    of AtomConfig)
    """

    # TODO: Consider how wraparound for periodic objects will work
    # TODO: Consider how filtering by element will work - should config array include element type?
    # TODO: Consider storage of weakref to each atom - fine for configuration but requires a complete atom object for each atom in every configuration in a trajectory

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
            return Trajectory(*self.filter_configs_by_time(item))

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

    # TODO: Refactor how filter deals with end=None
    def filter_configs_by_time(self, start, end=None):

        """
        Returns configurations with times in half open interval defined by start
        and end
        """
        if end is None:
            if start in self.times:
                return self.configurations[(self.times == start)]
            else:
                raise ValueError("Value is not in self.times")
        return self.configurations[(self.times >= start) & (self.times < end)]

    def __len__(self):
        return sum([len(config) for config in self.data['configuration']])


class DistanceData(object):

    """
    A container for calculating and storing separation distances determined from
    trajectories
    """

    def __init__(self, trajectory):
        self.data = np.array([(frame['time'],
            list(self._calculate_distances(
            frame['configuration'].atom_positions)))
            for frame in trajectory.data],
            dtype = [('time', 'float64'), ('distances', 'object')])

    @property
    def distances(self):
        return np.array([distance for distances in self.data['distances']
            for distance in distances])

    def _calculate_distances(self, positions):

        """
        Returns a generator of pairwise distances
        """

        # TODO: Something more pythonic
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                yield self._distance(positions[i], positions[j])

    # TODO: Consider if this will be extracted into a vector class
    def _distance(self,vec1,vec2):
        return np.linalg.norm(vec1-vec2)


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
        self.distance_data = DistanceData(trajectory)

        self.data = self._histogram_distance_data(self.r_axis)

        if self.time_axis:
            self._rebin_time()

    @property
    def histograms(self):
        return self.data['histogram']

    @property
    def times(self):
        return self.data['time']

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
            for frame in self.distance_data.data],
            dtype = [('time', 'float64'),
            ('histogram', 'object')])

    # TODO: Generalise to _rebin
    def _rebin_time(self):

        """
        Rebins histogram data

        Determines the upper and lower bounds and the centers of each bin.
        These are used to rebin the histograms.
        """

        n_bins = self._calc_n_bins(self.time_axis)
        bin_bounds = np.linspace(self.time_axis[0], self.time_axis[1], n_bins+1)
        bin_centers = bin_bounds[0:-1] + (bin_bounds[1] - bin_bounds[0]) / 2.

        # TODO: Refactor to remove empty array creation

        # rebin = np.array([])
        # for i in range(len(bin_bounds) - 1):
        #     rebin = np.append(rebin,
        #         self._sum_histograms(*self[bin_bounds[i]:bin_bounds[i+1]]))

        self.data = np.array([(bin_centers[i],
            self._sum_histograms(*self[bin_bounds[i]:bin_bounds[i + 1]]))
            for i in range(len(bin_bounds) - 1)],
            dtype = [('time','float64'),
            ('histogram','object')])

    def _sum_histograms(self, *histograms):

        """
        Returns the sum of histograms, assuming that they have identical bins
        """

        # TODO: Test for identical bins and return error if not
        bin_bounds = histograms[0][1]
        histogram_sum = [np.sum(
            [histogram[0] for histogram in histograms], axis=0),
            bin_bounds]

        return histogram_sum

    def _calc_n_bins(self, axis_bins):
        return int((axis_bins[1] - axis_bins[0]) / axis_bins[2])

    def filter_histogram_by_time(self, start, end):

        """
        Returns histograms with times in half open interval defined by start
        and end
        """

        return self.histograms[(self.times >= start) & (self.times < end)]

    def __getitem__(self, item):

        """
        Indexing and slicing is relative to time

        Returns histograms with times in half open interval defined by start
        and end
        """

        try:
            return self.filter_histogram_by_time(item.start, item.stop)
        except AttributeError:
            raise ValueError("Trajectory can only be sliced, not indexed")
