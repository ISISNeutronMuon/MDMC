"""Module for configuration, trajectory and histogram containers

AUTHOR :    Thomas Farmer        START DATE :    2018-5-29 23:44:11"""

import numpy as np

# TODO: SORT OUT EFFECTS OF PBC


class Configuration(object):

    """
    A Configuration stores atoms and their positions and velocities (in the form
    of AtomConfig)
    """

    # TODO: Consider how wraparound for periodic objects will work
    # TODO: Consider how filtering by element will work - should config array include element type?
    # TODO: Consider storage of weakref to each atom - fine for configuration but requires a complete atom object for each atom in every configuration in a trajectory

    def __init__(self, *structural_units):
        self.structures_list = list(structural_units)
        self.data = self.create_config_array(*structural_units)
        self.element_set = set(self.element_list)

    @property
    def atom_list(self):

        return self.data['atom']

    @property
    def atom_positions(self):

        return self.data['position']

    @property
    def atom_velocities(self):

        return self.data['velocity']

    @property
    def element_list(self):

        return [atom.element for atom in self.atom_list]

    @property
    def molecule_list(self):

        return self.filter_structures(lambda x: x.structure_type == 'Molecule')

    def create_config_array(self, *structural_units):

        return np.array([(atom, atom.position, atom.velocity)
            for unit in structural_units
            for atom in unit.atom_list],
            dtype=[('atom','object'),
            ('position','object'),
            ('velocity','object')])

    def add_structural_units(self, *structural_units):

        self.structures_list.extend(structural_units)
        self.data = np.append(
            self.data,self.create_config_array(*structural_units))

    def __add__(self, configuration):

        """
        Returns a new configuration from the sum of configurations
        """

        structures_list = self.structures_list + configuration.structures_list

        return self.__class__(*structures_list)

    def __sub__(self, configuration):

        """
        Returns a new configuration from the difference of configurations
        """

        raise NotImplementedError

    def __len__(self):

        return len(self.atom_list)

    def __getitem__(self, item):

        """
        Returns:
        A numpy void containing a slice from the data. The same fields can be
        accessed with 'atom', 'position', and 'velocity'.
        """

        return self.data[item]

    def filter_structures(self, predicate):

        """
        Filters the list of structural units using the predicate

        Arguments:
        predicate: a boolean valued function which can be applied to structural
        units
        """

        return filter(predicate, self.structures_list)

    def filter_atoms(self, predicate):

        """
        Filters the list of atoms using the predicate

        Arguments:
        predicate: a boolean valued function which can be applied to structural
        units
        """

        return filter(predicate, self.atom_list)

    def filter_by_element(self, element):

        """
        Filter the configuration using an element

        Arguments:
        element: elemental symbol of the same format as is used for creating
                 atoms
        """

        return self.filter_atoms(lambda x: x.element == element)

    # TODO: Implement
    def scale(self, factor, vectors='positions'):

        """
        Scales either atom positions (default) or velocities by a factor

        Arguments:
        factor: float by which the vector is scaled
        vectors: 'positions' (default) or 'velocities' of the atoms
        """

        raise NotImplementedError

class TemporalConfiguration(Configuration):

    def __init__(self, time, *structural_units):
        super(TemporalConfiguration, self).__init__(*structural_units)
        self.time = time

    def __add__(self, configuration):

        # TODO: Add warning if configurations don't all have the same time
        time = np.mean([self.time, configuration.time])

        structures_list = self.structures_list + configuration.structures_list

        return self.__class__(time, *structures_list)


class Trajectory(object):

    """
    A Trajectory is a collection of TimedConfigurations

    Attributes:
    data: an ordered array of frames, times and TimedConfigurations
    configurations: TimedConfigurations
    """

    def __init__(self, *configurations):

        self.data = configurations

    @property
    def data(self):

        return self._data

    @data.setter
    def data(self, configurations):

    # TODO: Test that all configurations have the same atoms

        self._data = np.array(
            [(i, config.time, config) for
            i, config in enumerate(configurations, 1)],
            dtype = [('frame', 'int64'),
            ('time', 'float64'),
            ('configuration', 'object')])

    def __getitem__(self, item):

        """
        Indexing and slicing is relative to frames
        """
        if type(item) == int:
            return self.__class__(self.configurations[item])
        else:
            return self.__class__(*self.configurations[item])

    @property
    def frames(self):

        return self.data['frame']

    @property
    def times(self):

        return self.data['time']

    @property
    def atoms(self):

        """
        Returns:
        Atoms from the frame 0 configuration
        """

        return self.data['configuration'][0].atom_list

    @property
    def element_set(self):

        """
        Returns:
        Set of elements from the frame 0 configuration
        """

        return self.data['configuration'][0].element_set

    @property
    def element_list(self):

        """
        Returns:
        List of elements from the frame 0 configuration
        """

        return self.data['configuration'][0].element_list

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

    def filter_by_time(self, start, end=None):

        """
        Returns:
        Trajectory with times in half open interval defined by start and end
        """

        if end is None:
            try:
                return self.__class__(*self.configurations[
                    (self.times == start)])
            except IndexError:
                raise ValueError("Start is not in self.times")
        return self.__class__(*self.configurations[
            (self.times >= start) & (self.times < end)])

    def __len__(self):

        return sum([len(config) for config in self.data['configuration']])


class DistanceData(object):

    """
    A container for calculating and storing separation distances determined from
    trajectories

    Attributes:
    data: an ordered array of times and atomic separations from each
    TimedConfiguration
    distances: all atomic separations
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

        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                yield self._distance(positions[i], positions[j])

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
